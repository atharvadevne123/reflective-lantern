"""FastAPI application with /predict, /health, and /metrics endpoints."""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db, init_db
from app.exceptions import (
    FeatureExtractionError,
    ModelNotLoadedError,
    register_exception_handlers,
)
from app.features import (
    CARRIERS,
    ROUTE_TYPES,
    build_feature_pipeline,
    generate_synthetic_data,
    prepare_X,
)
from app.middleware import RateLimitMiddleware
from app.model import load_model, train_model
from app.monitoring import log_prediction, run_drift_check, seed_reference_buffer

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"
    if os.getenv("LOG_LEVEL") == "DEBUG"
    else "%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_VERSION = "1.0.0"
FEATURE_PIPELINE_PATH = os.getenv("FEATURE_PIPELINE_PATH", "feature_pipeline.joblib")
METRICS_PATH = os.getenv("METRICS_PATH", "metrics.json")

_model = None
_feat_pipe = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _feat_pipe
    init_db()

    if Path(FEATURE_PIPELINE_PATH).exists():
        _feat_pipe = joblib.load(FEATURE_PIPELINE_PATH)
        logger.info("Feature pipeline loaded from disk")
    else:
        df = generate_synthetic_data(n=2000)
        _feat_pipe = build_feature_pipeline()
        X = prepare_X(df, _feat_pipe, fit=True)
        y = df["delivery_minutes"].values
        _model, _ = train_model(X, y)
        joblib.dump(_feat_pipe, FEATURE_PIPELINE_PATH)
        seed_reference_buffer(df.sample(200).to_dict("records"))

    _model = load_model()
    logger.info("Model loaded — version %s", MODEL_VERSION)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Logistics-Flow API",
    description="Last-mile delivery time prediction with drift monitoring.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

settings = get_settings()

register_exception_handlers(app)
app.add_middleware(RateLimitMiddleware, limit=settings.rate_limit_per_minute)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Attach a unique request-id to each request for tracing."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.1f}"
    return response


# ── Request / Response models ────────────────────────────────────────────────


class PredictRequest(BaseModel):
    """Input payload for a single delivery prediction."""

    carrier: str = Field(..., examples=["DHL"])
    distance_km: float = Field(..., gt=0, le=5000, examples=[42.5])
    weight_kg: float = Field(..., gt=0, le=100, examples=[3.2])
    route_type: str = Field(..., examples=["urban"])
    hour_of_day: int = Field(..., ge=0, le=23, examples=[14])
    day_of_week: int = Field(..., ge=0, le=6, examples=[2])

    @field_validator("carrier")
    @classmethod
    def carrier_must_be_known(cls, v: str) -> str:
        if v not in CARRIERS:
            raise ValueError(f"carrier must be one of {CARRIERS}")
        return v

    @field_validator("route_type")
    @classmethod
    def route_must_be_known(cls, v: str) -> str:
        if v not in ROUTE_TYPES:
            raise ValueError(f"route_type must be one of {ROUTE_TYPES}")
        return v


class PredictResponse(BaseModel):
    predicted_minutes: float
    predicted_hours: float
    confidence: float
    model_version: str
    request_id: str


class BatchPredictRequest(BaseModel):
    """Up to 100 shipments scored in a single call."""

    shipments: list[PredictRequest] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded: bool


class MetricsResponse(BaseModel):
    rmse_mean: float | None
    r2_mean: float | None
    n_features: int | None
    n_samples: int | None
    model_version: str


# ── Helpers ──────────────────────────────────────────────────────────────────


def _ensemble_confidence(X: np.ndarray, predicted: float) -> float:
    """Derive confidence from agreement between ensemble sub-estimators.

    Lower spread across the individual regressors means the ensemble agrees,
    which we map to a higher confidence score in [0, 1].
    """
    try:
        scaler = _model.named_steps["scaler"]
        voting = _model.named_steps["model"]
        X_scaled = scaler.transform(X)
        member_preds = [est.predict(X_scaled)[0] for est in voting.estimators_]
    except Exception:  # pragma: no cover - defensive
        logger.debug("Could not compute ensemble spread", exc_info=True)
        return 0.5

    if len(member_preds) < 2 or predicted <= 0:
        return 0.5

    spread = float(np.std(member_preds))
    # Normalise spread against the prediction magnitude
    return round(float(max(0.0, min(1.0, 1.0 - spread / (abs(predicted) + 1e-6)))), 4)


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.post(
    "/api/v1/predict",
    response_model=PredictResponse,
    summary="Predict delivery time",
    description="Returns estimated delivery duration in minutes and hours.",
)
async def predict(
    payload: PredictRequest,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
):
    if _model is None or _feat_pipe is None:
        raise ModelNotLoadedError

    import pandas as pd

    row = pd.DataFrame([payload.model_dump()])
    try:
        X = prepare_X(row, _feat_pipe, fit=False)
    except Exception as exc:
        logger.exception("Feature extraction failed")
        raise FeatureExtractionError(str(exc)) from exc

    predicted = float(_model.predict(X)[0])
    confidence = _ensemble_confidence(X, predicted)

    log_prediction(
        db,
        carrier=payload.carrier,
        distance_km=payload.distance_km,
        weight_kg=payload.weight_kg,
        route_type=payload.route_type,
        hour_of_day=payload.hour_of_day,
        day_of_week=payload.day_of_week,
        predicted_minutes=predicted,
        confidence=confidence,
        model_version=MODEL_VERSION,
    )

    return PredictResponse(
        predicted_minutes=round(predicted, 2),
        predicted_hours=round(predicted / 60, 3),
        confidence=round(confidence, 4),
        model_version=MODEL_VERSION,
        request_id=getattr(request.state, "request_id", "n/a"),
    )


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health():
    return HealthResponse(
        status="healthy" if _model is not None else "degraded",
        model_version=MODEL_VERSION,
        model_loaded=_model is not None,
    )


@app.get(
    "/api/v1/metrics",
    response_model=MetricsResponse,
    summary="Model performance metrics",
    description="Returns last-computed cross-validation metrics.",
)
async def metrics():
    data = json.loads(Path(METRICS_PATH).read_text()) if Path(METRICS_PATH).exists() else {}
    return MetricsResponse(
        rmse_mean=data.get("rmse_mean"),
        r2_mean=data.get("r2_mean"),
        n_features=data.get("n_features"),
        n_samples=data.get("n_samples"),
        model_version=data.get("model_version", MODEL_VERSION),
    )


@app.get(
    "/api/v1/drift",
    summary="Run drift check",
    description="Compares recent predictions against reference distribution.",
)
async def drift(db: Annotated[Session, Depends(get_db)]):
    return run_drift_check(db)


@app.post(
    "/api/v1/predict/batch",
    response_model=BatchPredictResponse,
    summary="Predict delivery times in bulk",
    description="Scores up to 100 shipments in one request.",
)
async def predict_batch(
    payload: BatchPredictRequest,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
):
    if _model is None or _feat_pipe is None:
        raise ModelNotLoadedError

    import pandas as pd

    rows = pd.DataFrame([s.model_dump() for s in payload.shipments])
    try:
        X = prepare_X(rows, _feat_pipe, fit=False)
    except Exception as exc:
        logger.exception("Batch feature extraction failed")
        raise FeatureExtractionError(str(exc)) from exc

    raw = _model.predict(X)
    request_id = getattr(request.state, "request_id", "n/a")

    results: list[PredictResponse] = []
    for shipment, predicted in zip(payload.shipments, raw, strict=True):
        predicted = float(predicted)
        confidence = _ensemble_confidence(X, predicted)
        log_prediction(
            db,
            carrier=shipment.carrier,
            distance_km=shipment.distance_km,
            weight_kg=shipment.weight_kg,
            route_type=shipment.route_type,
            hour_of_day=shipment.hour_of_day,
            day_of_week=shipment.day_of_week,
            predicted_minutes=predicted,
            confidence=confidence,
            model_version=MODEL_VERSION,
        )
        results.append(
            PredictResponse(
                predicted_minutes=round(predicted, 2),
                predicted_hours=round(predicted / 60, 3),
                confidence=round(confidence, 4),
                model_version=MODEL_VERSION,
                request_id=request_id,
            )
        )

    return BatchPredictResponse(predictions=results, count=len(results))
