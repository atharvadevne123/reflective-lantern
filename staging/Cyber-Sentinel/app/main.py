"""Cyber-Sentinel FastAPI application for real-time intrusion detection."""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import __version__
from app.cache import cache_prediction, cache_stats, cached_predict
from app.database import Prediction, get_db, init_db
from app.features import build_feature_vector
from app.model import get_feature_importance, predict, train_model
from app.monitoring import (
    check_alerts,
    get_feature_drift_summary,
    log_prediction,
    prediction_health,
)
from app.retrieval import get_index

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(
    title="Cyber-Sentinel",
    description=(
        "Real-time network intrusion and cyber attack detection API using "
        "XGBoost-LightGBM-RandomForest ensemble with KS-drift monitoring, "
        "FAISS pattern matching, and Airflow retraining pipelines."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next: Any) -> Any:
    """Attach a correlation ID to every request for traceability."""
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    request.state.correlation_id = correlation_id
    start = time.monotonic()
    response = await call_next(request)
    elapsed = round((time.monotonic() - start) * 1000, 2)
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Response-Time-Ms"] = str(elapsed)
    logger.info(
        "path=%s method=%s status=%d ms=%s cid=%s",
        request.url.path,
        request.method,
        response.status_code,
        elapsed,
        correlation_id,
    )
    return response


@app.on_event("startup")
async def startup_event() -> None:
    """Initialise the database on startup."""
    try:
        init_db()
    except Exception as exc:
        logger.error("DB init failed at startup: %s", exc)


class NetworkEventRequest(BaseModel):
    """Payload for a network event to be analysed."""

    protocol: str = Field(default="tcp", description="Network protocol (tcp, udp, icmp)")
    payload_bytes: int = Field(default=0, ge=0, description="Total payload size in bytes")
    duration_ms: float = Field(default=0.0, ge=0.0, description="Connection duration (ms)")
    src_bytes: int = Field(default=0, ge=0, description="Bytes from source host")
    dst_bytes: int = Field(default=0, ge=0, description="Bytes from destination host")
    flags: str = Field(default="", description="TCP flags string (e.g. 'SA', 'FIN')")
    src_ip: str = Field(default="", description="Source IP address")
    dst_ip: str = Field(default="", description="Destination IP address")
    wrong_fragment: int = Field(default=0, ge=0)
    urgent: int = Field(default=0, ge=0)
    hot: int = Field(default=0, ge=0)
    num_failed_logins: int = Field(default=0, ge=0)
    logged_in: int = Field(default=0, ge=0, le=1)
    num_compromised: int = Field(default=0, ge=0)
    root_shell: int = Field(default=0, ge=0, le=1)
    su_attempted: int = Field(default=0, ge=0, le=1)
    num_root: int = Field(default=0, ge=0)
    num_file_creations: int = Field(default=0, ge=0)
    count: int = Field(default=0, ge=0)
    srv_count: int = Field(default=0, ge=0)
    serror_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    rerror_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    same_srv_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    diff_srv_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class TrainRequest(BaseModel):
    """Optional payload for the train endpoint."""

    n_samples: int = Field(default=2000, ge=100, le=100000)


@app.get("/health", tags=["System"])
async def health() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok", "version": __version__}


@app.get("/version", tags=["System"])
async def version() -> dict[str, str]:
    """Return the current API version."""
    return {"version": __version__}


@app.post("/train", tags=["Model"])
async def train(
    request: TrainRequest = TrainRequest(),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Train the ensemble model on synthetic or provided data.

    Returns training metrics including CV F1 and per-class precision/recall.
    """
    try:
        metrics = train_model()
        logger.info("Training complete: %s", metrics)
        return {"success": True, "metrics": metrics}
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict", tags=["Inference"])
async def predict_endpoint(
    event: NetworkEventRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Classify a network event as normal or attack.

    Also performs FAISS pattern matching and logs the result for drift tracking.
    """
    try:
        features = build_feature_vector(event.model_dump())
        result = cached_predict(features)
        if result is None:
            result = predict(features)
            cache_prediction(features, result)
        log_prediction(result, features)

        index = get_index()
        similar_patterns = index.search(features[:index.dim], k=3)
        result["similar_patterns"] = similar_patterns

        try:
            db_pred = Prediction(
                label=result["label"],
                confidence=result["confidence"],
                attack_type=result.get("attack_type"),
                model_version=result.get("model_version"),
            )
            db.add(db_pred)
            db.commit()
        except Exception as db_exc:
            logger.warning("DB write failed: %s", db_exc)
            db.rollback()

        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/metrics", tags=["Monitoring"])
async def metrics() -> dict[str, Any]:
    """Return prediction health and feature drift summary."""
    health = prediction_health()
    drift = get_feature_drift_summary()
    importance = {}
    try:
        importance = get_feature_importance()
    except RuntimeError:
        pass
    return {
        "prediction_health": health,
        "drift_summary": drift,
        "feature_importance": importance,
    }


@app.get("/drift", tags=["Monitoring"])
async def drift_check() -> dict[str, Any]:
    """Run drift detection and return any active alerts."""
    summary = get_feature_drift_summary()
    drift_list = list(summary.values()) if isinstance(summary, dict) else []
    alerts = check_alerts([d for d in drift_list if isinstance(d, dict)])
    return {"alerts": alerts, "drift_details": summary}


@app.get("/cache/stats", tags=["System"])
async def cache_statistics() -> dict[str, Any]:
    """Return prediction cache hit statistics."""
    return cache_stats()


@app.get("/feature-importance", tags=["Model"])
async def feature_importance() -> dict[str, Any]:
    """Return per-feature importance scores from the Random Forest base learner."""
    try:
        importance = get_feature_importance()
        return {"feature_importance": importance}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
