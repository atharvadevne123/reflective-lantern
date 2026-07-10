"""Forge-Guard FastAPI application — manufacturing defect prediction service."""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.features import engineer_single
from app.model import MODEL_VERSION, get_metrics, load_model, predict
from app.monitoring import defect_rate, log_prediction, run_drift_check
from app.schemas import BatchPredictionResponse, BatchSensorInput

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _model
    init_db()
    _model = load_model()
    logger.info("Forge-Guard ready — model version %s", MODEL_VERSION)
    yield


app = FastAPI(
    title="Forge-Guard",
    description="Real-time manufacturing defect prediction API with drift monitoring.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_RATE_WINDOW: dict[str, list[float]] = {}
RATE_LIMIT = int(os.getenv("RATE_LIMIT_RPM", "60"))


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next: Any) -> Any:
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = _RATE_WINDOW.setdefault(client_ip, [])
    _RATE_WINDOW[client_ip] = [t for t in window if now - t < 60]
    if len(_RATE_WINDOW[client_ip]) >= RATE_LIMIT:
        return JSONResponse(
            {"detail": "Rate limit exceeded"}, status_code=status.HTTP_429_TOO_MANY_REQUESTS
        )
    _RATE_WINDOW[client_ip].append(now)
    return await call_next(request)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next: Any) -> Any:
    cid = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    request.state.correlation_id = cid
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = cid
    return response


class SensorInput(BaseModel):
    """Validated payload for a single inference request."""

    temperature: float = Field(..., ge=-50, le=300, description="Process temperature in °C")
    pressure: float = Field(..., ge=0, le=200, description="Chamber pressure in bar")
    vibration: float = Field(..., ge=0, le=50, description="Vibration amplitude in mm/s")
    cycle_time: float = Field(..., ge=0.1, le=600, description="Cycle duration in seconds")
    tool_wear: float = Field(..., ge=0, le=500, description="Cumulative tool wear in µm")
    power_consumption: float = Field(..., ge=0, le=500, description="Energy draw in kW")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity %")

    @field_validator("temperature", "pressure", "vibration", "cycle_time", "power_consumption")
    @classmethod
    def must_be_finite(cls, v: float) -> float:
        import math
        if not math.isfinite(v):
            raise ValueError("Value must be finite")
        return v


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = OK, 1 = Defect")
    defect_probability: float
    model_version: str
    correlation_id: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class MetricsResponse(BaseModel):
    model_metrics: dict[str, Any]
    defect_rate_24h: dict[str, Any]
    drift_reports: list[dict[str, Any]]


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness probe endpoint."""
    return HealthResponse(
        status="healthy" if _model is not None else "degraded",
        model_loaded=_model is not None,
        model_version=MODEL_VERSION,
    )


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    summary="Predict manufacturing defect from sensor readings",
)
async def predict_defect(
    payload: SensorInput,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> PredictionResponse:
    """Run the ensemble model on a single sensor snapshot and return defect probability."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    sensor_dict = payload.model_dump()
    features = engineer_single(sensor_dict)

    label, prob = predict(_model, features)

    cid = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    log_prediction(
        db=db,
        sensor_data=sensor_dict,
        prediction=label,
        defect_probability=prob,
        model_version=MODEL_VERSION,
        correlation_id=cid,
    )
    logger.info("predict cid=%s label=%d prob=%.4f", cid, label, prob)

    return PredictionResponse(
        prediction=label,
        defect_probability=prob,
        model_version=MODEL_VERSION,
        correlation_id=cid,
    )


# Legacy alias kept for backwards compatibility with v0 clients
@app.post("/predict", response_model=PredictionResponse, tags=["inference"], deprecated=True)
async def predict_legacy(
    payload: SensorInput,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> PredictionResponse:
    return await predict_defect(payload, request, db)


@app.get(
    "/api/v1/metrics",
    response_model=MetricsResponse,
    tags=["ops"],
    summary="Model performance metrics and drift reports",
)
async def metrics(db: Annotated[Session, Depends(get_db)]) -> MetricsResponse:
    """Return training metrics, 24-hour defect rate, and latest drift check results."""
    drift = run_drift_check(db, model_version=MODEL_VERSION)
    return MetricsResponse(
        model_metrics=get_metrics(),
        defect_rate_24h=defect_rate(db),
        drift_reports=drift,
    )


@app.get("/metrics", tags=["ops"], deprecated=True)
async def metrics_legacy(db: Annotated[Session, Depends(get_db)]) -> MetricsResponse:
    return await metrics(db)


@app.get(
    "/api/v1/feature-importance",
    tags=["inference"],
    summary="Return XGBoost feature importances",
)
async def feature_importance_endpoint() -> dict[str, float]:
    """Return per-feature importance scores from the XGBoost sub-estimator."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    from app.model import feature_importance

    return feature_importance(_model)


@app.post(
    "/api/v1/predict/batch",
    tags=["inference"],
    summary="Batch predict defects for up to 100 sensor readings",
)
async def predict_batch(
    payload: BatchSensorInput,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> BatchPredictionResponse:
    """Run inference on multiple sensor readings in a single request."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for reading in payload.readings:
        feats = engineer_single(reading)
        label, prob = predict(_model, feats)
        log_prediction(db=db, sensor_data=reading, prediction=label,
                       defect_probability=prob, model_version=MODEL_VERSION)
        results.append({"prediction": label, "defect_probability": prob})

    return BatchPredictionResponse(
        predictions=results, count=len(results), model_version=MODEL_VERSION
    )
