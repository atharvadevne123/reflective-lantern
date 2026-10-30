"""Fleet-Mind FastAPI application — vehicle telemetry anomaly detection."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db, init_db
from app.features import RAW_FEATURES, build_feature_pipeline
from app.logging_config import configure_logging
from app.middleware import CorrelationIDMiddleware, RateLimitMiddleware
from app.model import MODEL_VERSION, get_metrics, load_model, predict, train_model
from app.monitoring import log_prediction, maintenance_rate, run_drift_check
from app.schemas import (
    BatchPredictionResponse,
    BatchTelemetryInput,
    DriftCheckResponse,
    HealthResponse,
    MetricsResponse,
    PredictionResponse,
    TelemetryInput,
)

settings = get_settings()
configure_logging(
    level=settings.log_level,
    json_output=settings.log_json,
)
logger = logging.getLogger(__name__)

_model: Any = None
_feature_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _model, _feature_pipeline
    init_db()
    _model = load_model()
    _feature_pipeline = build_feature_pipeline()
    # Fit on dummy data so scaler is initialised
    dummy_df = pd.DataFrame(
        data=[[0] * len(RAW_FEATURES)],
        columns=RAW_FEATURES,
    )
    _feature_pipeline.fit(dummy_df)
    logger.info("Fleet-Mind ready — model version %s", MODEL_VERSION)
    yield


TAGS = [
    {"name": "inference", "description": "Single and batch prediction endpoints."},
    {"name": "ops", "description": "Health, metrics, drift, and retraining endpoints."},
]

app = FastAPI(
    title="Fleet-Mind",
    description=(
        "Real-time fleet vehicle telemetry anomaly detection and predictive maintenance. "
        "Send 10 sensor readings per telemetry cycle and receive maintenance flags, "
        "anomaly type classification, and remaining useful life estimates."
    ),
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=TAGS,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(CorrelationIDMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_requests)


def _raw_features(reading: TelemetryInput) -> dict[str, float]:
    return {f: getattr(reading, f) for f in RAW_FEATURES}


# ── Inference ──────────────────────────────────────────────────────────────


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    summary="Predict maintenance need for a single vehicle",
)
def predict_single(
    body: TelemetryInput,
    db: Annotated[Session, Depends(get_db)],
    request: Request,
) -> PredictionResponse:
    """Analyse one set of telemetry readings and return a maintenance prediction."""
    if _model is None or _feature_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    raw = _raw_features(body)
    try:
        df = pd.DataFrame([raw])
        features = _feature_pipeline.transform(df)
        result = predict(_model, features, raw)
    except Exception as exc:
        logger.error("Prediction failed for vehicle %s: %s", body.vehicle_id, exc)
        raise HTTPException(status_code=500, detail="Prediction error") from exc

    log_prediction(db, body.vehicle_id, raw, result)
    return PredictionResponse(vehicle_id=body.vehicle_id, **result)


@app.post(
    "/api/v1/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["inference"],
    summary="Batch predict maintenance for up to 100 vehicles",
)
def predict_batch(
    body: BatchTelemetryInput,
    db: Annotated[Session, Depends(get_db)],
) -> BatchPredictionResponse:
    """Analyse a batch of telemetry readings."""
    if _model is None or _feature_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    results: list[PredictionResponse] = []
    for reading in body.readings:
        raw = _raw_features(reading)
        try:
            df = pd.DataFrame([raw])
            features = _feature_pipeline.transform(df)
            result = predict(_model, features, raw)
        except Exception as exc:
            logger.error("Batch prediction failed for %s: %s", reading.vehicle_id, exc)
            continue
        log_prediction(db, reading.vehicle_id, raw, result)
        results.append(PredictionResponse(vehicle_id=reading.vehicle_id, **result))

    maintenance_flagged = sum(1 for r in results if r.maintenance_needed)
    return BatchPredictionResponse(
        results=results,
        total=len(results),
        maintenance_flagged=maintenance_flagged,
    )


# ── Ops ───────────────────────────────────────────────────────────────────


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["ops"],
    summary="Health check",
)
def health(db: Annotated[Session, Depends(get_db)]) -> HealthResponse:
    """Return service liveness and model status."""
    db_ok = True
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
    except Exception:
        db_ok = False
    return HealthResponse(
        status="ok" if _model is not None else "degraded",
        model_version=MODEL_VERSION,
        db_ok=db_ok,
    )


@app.get(
    "/api/v1/metrics",
    response_model=MetricsResponse,
    tags=["ops"],
    summary="Return model performance metrics",
)
def metrics(db: Annotated[Session, Depends(get_db)]) -> MetricsResponse:
    """Return latest model training metrics and live maintenance rate."""
    data = get_metrics()
    data["maintenance_rate"] = maintenance_rate(db)
    return MetricsResponse(**data)


@app.get(
    "/api/v1/drift",
    response_model=DriftCheckResponse,
    tags=["ops"],
    summary="Run KS-test drift analysis on recent telemetry",
)
def drift_check(db: Annotated[Session, Depends(get_db)]) -> DriftCheckResponse:
    """Compare recent telemetry distributions against training baseline."""
    reports = run_drift_check(db)
    drifted = sum(1 for r in reports if r.get("drift_detected"))
    return DriftCheckResponse(
        checked_features=len(reports),
        drifted_features=drifted,
        reports=reports,
    )


@app.post(
    "/api/v1/retrain",
    tags=["ops"],
    summary="Trigger model retraining",
)
def retrain() -> dict[str, Any]:
    """Retrain the model on fresh synthetic data and reload into memory."""
    global _model
    try:
        new_metrics = train_model()
        _model = load_model()
        logger.info("Model retrained: %s", new_metrics)
        return {"status": "ok", "metrics": new_metrics}
    except Exception as exc:
        logger.error("Retraining failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Retraining failed: {exc}") from exc


@app.get("/api/v1/version", tags=["ops"], summary="Return API and model version")
def version() -> dict[str, str]:
    """Return current API and model version strings."""
    return {"api_version": "1.0.0", "model_version": MODEL_VERSION}
