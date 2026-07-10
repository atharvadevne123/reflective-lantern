"""FastAPI application — /api/v1/predict, /api/v1/metrics, /api/v1/drift, /health."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.features import FEATURE_COLUMNS, build_feature_vector
from app.logging_config import configure_logging
from app.middleware import rate_limit_middleware
from app.model import MODEL_VERSION, load_model
from app.model import predict as model_predict
from app.monitoring import (
    compute_drift,
    get_drift_summary,
    get_recent_predictions,
    log_drift,
    log_prediction,
)

configure_logging()
logger = logging.getLogger(__name__)

_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    init_db()
    _models.update(load_model())
    logger.info("Traffic-Pulse started model_version=%s", MODEL_VERSION)
    yield
    _models.clear()


TAGS_METADATA = [
    {"name": "prediction", "description": "Congestion level predictions (single and batch)."},
    {"name": "monitoring", "description": "Drift detection and prediction statistics."},
    {"name": "system", "description": "Health and model metadata."},
]

app = FastAPI(
    title="Traffic-Pulse",
    description=(
        "Real-time urban traffic congestion prediction and incident detection API "
        "using an XGBoost-LightGBM ensemble with KS-test drift monitoring."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
    openapi_tags=TAGS_METADATA,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


app.add_middleware(GZipMiddleware, minimum_size=1024)

app.middleware("http")(rate_limit_middleware)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next: Any) -> Any:
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
    return response


# ── Request / Response schemas ───────────────────────────────────────────────


class TrafficInput(BaseModel):
    route_id: str = Field(..., min_length=1, max_length=64)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(default=6, ge=1, le=12)
    vehicle_count: int = Field(..., ge=0, le=10_000)
    avg_speed_kmh: float = Field(..., ge=0.0, le=200.0)
    road_type: str = Field(default="arterial")
    incident_count: int = Field(default=0, ge=0, le=100)
    temperature_celsius: float = Field(default=20.0, ge=-30.0, le=50.0)
    is_raining: int = Field(default=0, ge=0, le=1)
    lag_1h: float | None = None
    lag_2h: float | None = None
    lag_4h: float | None = None
    rolling_mean_6h: float | None = None
    rolling_std_6h: float | None = None
    rolling_mean_24h: float | None = None

    @field_validator("road_type")
    @classmethod
    def validate_road_type(cls, v: str) -> str:
        allowed = {"highway", "arterial", "collector", "local", "expressway"}
        if v.lower() not in allowed:
            msg = f"road_type must be one of {sorted(allowed)}"
            raise ValueError(msg)
        return v.lower()


class PredictionResponse(BaseModel):
    route_id: str
    congestion_level: int
    congestion_label: str
    congestion_probability: float
    class_probabilities: dict[str, float]
    incident_score: float
    model_version: str


class DriftRequest(BaseModel):
    feature_name: str
    reference: list[float] = Field(..., min_length=10)
    current: list[float] = Field(..., min_length=10)


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health", tags=["system"], summary="Liveness probe")
def health() -> dict[str, Any]:
    """Liveness probe — returns model version and loaded ensemble members."""
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "models_loaded": list(_models.keys()),
    }


@app.get("/api/v1/metrics", tags=["monitoring"], summary="Monitoring snapshot")
def metrics(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Prediction statistics and any active drift alerts."""
    predictions = get_recent_predictions(db, limit=200)
    drift_logs = get_drift_summary(db, limit=50)

    level_counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    for p in predictions:
        level_counts[p.congestion_level] = level_counts.get(p.congestion_level, 0) + 1

    metrics_path = Path("metrics.json")
    training_metrics: dict[str, Any] = (
        json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    )

    return {
        "total_predictions": len(predictions),
        "congestion_distribution": level_counts,
        "drift_alerts": [d for d in drift_logs if d["drift_detected"]],
        "training_metrics": training_metrics,
        "model_version": MODEL_VERSION,
    }


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    summary="Predict congestion for one route segment",
)
def predict(
    payload: TrafficInput,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """Predict congestion level (0-3) for the given route and conditions."""
    if not _models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    raw = payload.model_dump()
    features = build_feature_vector(raw)
    df = pd.DataFrame([features])

    result = model_predict(_models, df)
    log_prediction(
        db,
        hour=payload.hour,
        day_of_week=payload.day_of_week,
        route_id=payload.route_id,
        road_type=payload.road_type,
        prediction=result,
        features=features,
    )
    return PredictionResponse(route_id=payload.route_id, **result)


@app.post("/api/v1/drift", tags=["monitoring"], summary="Run KS drift test on a feature")
def check_drift(
    payload: DriftRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """KS-test drift detection between a reference and current feature window."""
    drift_result = compute_drift(payload.reference, payload.current)
    log_drift(db, payload.feature_name, drift_result)
    return {"feature": payload.feature_name, **drift_result}


@app.post(
    "/api/v1/predict/batch",
    tags=["prediction"],
    summary="Predict congestion for up to 100 segments",
)
def predict_batch(
    payloads: list[TrafficInput],
    db: Session = Depends(get_db),
) -> list[PredictionResponse]:
    """Predict congestion levels for up to 100 route segments in one call."""
    if not _models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    if len(payloads) > 100:
        raise HTTPException(status_code=422, detail="Batch size limited to 100")

    responses: list[PredictionResponse] = []
    for payload in payloads:
        features = build_feature_vector(payload.model_dump())
        result = model_predict(_models, pd.DataFrame([features]))
        log_prediction(
            db,
            hour=payload.hour,
            day_of_week=payload.day_of_week,
            route_id=payload.route_id,
            road_type=payload.road_type,
            prediction=result,
            features=features,
        )
        responses.append(PredictionResponse(route_id=payload.route_id, **result))
    return responses


@app.get("/api/v1/model-info", tags=["system"], summary="Model metadata and feature list")
def model_info() -> dict[str, Any]:
    """Model metadata: version, ensemble members, and feature list."""
    return {
        "model_version": MODEL_VERSION,
        "ensemble_members": list(_models.keys()),
        "n_features": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "congestion_labels": {"0": "free", "1": "moderate", "2": "congested", "3": "severe"},
    }

@app.get("/api/v1/routes/{route_id}/history", tags=["prediction"], summary="Recent predictions for a route")
def route_history(
    route_id: str,
    limit: int = 20,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Return the most recent logged predictions for one route segment."""
    from app.database import PredictionLog

    limit = max(1, min(limit, 200))
    rows = (
        db.query(PredictionLog)
        .filter(PredictionLog.route_id == route_id)
        .order_by(PredictionLog.timestamp.desc())
        .limit(limit)
        .all()
    )
    return {
        "route_id": route_id,
        "count": len(rows),
        "predictions": [
            {
                "timestamp": r.timestamp.isoformat(),
                "hour": r.hour,
                "congestion_level": r.congestion_level,
                "congestion_prob": r.congestion_prob,
                "incident_score": r.incident_score,
                "model_version": r.model_version,
            }
            for r in rows
        ],
    }
