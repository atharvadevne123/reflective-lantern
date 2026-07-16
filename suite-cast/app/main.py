"""Suite-Cast FastAPI application — hotel demand forecasting and dynamic pricing."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .config import settings
from .database import Prediction, create_tables, get_db
from .model import generate_sample_data, load_models, predict_demand
from .monitoring import (
    compute_drift,
    get_prediction_stats,
    log_prediction,
)

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format=(
        '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}'
    ),
)
logger = logging.getLogger(__name__)

_xgb_model: Any = None
_lgbm_model: Any = None
_model_metrics: dict[str, Any] = {}
_reference_scores: list[float] = []
_started_at: float = time.monotonic()

# In-memory rate-limit windows keyed by client IP
_request_windows: dict[str, list[float]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    """Startup: load models and build reference score distribution."""
    global _xgb_model, _lgbm_model, _model_metrics, _reference_scores  # noqa: PLW0603

    create_tables()
    _xgb_model, _lgbm_model, _model_metrics = load_models()

    # Build reference distribution over synthetic data for drift baseline
    x_ref, _ = generate_sample_data(500)
    xgb_probs: np.ndarray = _xgb_model.predict_proba(x_ref)[:, 1]
    if _lgbm_model is not None:
        lgbm_probs: np.ndarray = _lgbm_model.predict_proba(x_ref)[:, 1]
        _reference_scores = list((xgb_probs + lgbm_probs) / 2.0)
    else:
        _reference_scores = xgb_probs.tolist()

    logger.info(
        "Suite-Cast ready — ensemble AUC=%.4f ref_samples=%d",
        _model_metrics.get("auc_mean", 0.0),
        len(_reference_scores),
    )
    yield


app = FastAPI(
    title="Suite-Cast",
    description=(
        "Hotel booking demand forecasting and dynamic pricing API. "
        "Uses an XGBoost-LightGBM ensemble trained on historical booking patterns "
        "to predict occupancy demand and suggest optimal room rates."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next: Any) -> Any:
    """Sliding-window rate limiter keyed by client IP."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = [t for t in _request_windows.get(client_ip, []) if now - t < 60.0]
    if len(window) >= settings.rate_limit_per_minute:
        return JSONResponse({"detail": "Rate limit exceeded — try again in 60 s"}, status_code=429)
    window.append(now)
    _request_windows[client_ip] = window
    return await call_next(request)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next: Any) -> Any:
    """Propagate or generate a correlation ID for request tracing."""
    cid = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = cid
    return response


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class BookingRequest(BaseModel):
    """Input payload for the /predict endpoint.

    All monetary fields are in USD. Rates must be positive.
    """

    lead_time: int = Field(
        ...,
        ge=0,
        le=365,
        description="Days between booking date and check-in date.",
        examples=[14],
    )
    length_of_stay: int = Field(
        ..., ge=1, le=30, description="Number of nights booked.", examples=[3]
    )
    guests_count: int = Field(..., ge=1, le=10, examples=[2])
    checkin_month: int = Field(
        ..., ge=1, le=12, description="Calendar month of check-in (1=Jan).", examples=[7]
    )
    checkin_dayofweek: int = Field(
        ..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun).", examples=[4]
    )
    is_weekend: int = Field(
        ..., ge=0, le=1, description="1 if check-in falls on a weekend.", examples=[1]
    )
    current_occ_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Hotel's current overall occupancy rate.", examples=[0.72]
    )
    prev_year_occ_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Same-period occupancy rate from prior year.",
        examples=[0.68],
    )
    room_rate: float = Field(
        ..., ge=10.0, le=5000.0, description="Proposed room rate (USD).", examples=[189.0]
    )
    competitor_avg_rate: float = Field(
        ...,
        ge=10.0,
        le=5000.0,
        description="Competitor average rate for equivalent rooms (USD).",
        examples=[175.0],
    )
    special_event: int = Field(
        0, ge=0, le=1, description="1 if a local high-demand event is scheduled.", examples=[0]
    )
    room_type: str = Field(
        "standard",
        pattern="^(standard|deluxe|suite)$",
        description="Room category.",
        examples=["deluxe"],
    )
    booking_channel: str = Field(
        "online",
        pattern="^(direct|online|ota)$",
        description="Booking acquisition channel.",
        examples=["online"],
    )


class PredictionResponse(BaseModel):
    """Response payload from the /predict endpoint."""

    request_id: str = Field(..., description="Unique identifier for this prediction record.")
    demand_score: float = Field(..., description="Predicted booking demand probability (0–1).")
    demand_tier: str = Field(..., description="Demand tier: low / medium / high.")
    suggested_rate: float = Field(..., description="Dynamically priced room rate suggestion (USD).")
    model_version: str = Field(..., description="Model version that produced this prediction.")


class HealthResponse(BaseModel):
    """Response payload from the /health endpoint."""

    status: str
    model_loaded: bool
    model_version: str
    auc_mean: float | None
    environment: str
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post(
    f"{settings.api_prefix}/predict",
    response_model=PredictionResponse,
    summary="Predict hotel booking demand",
    description=(
        "Given booking request parameters, returns a demand probability score (0–1), "
        "demand tier classification, and a dynamically priced room rate suggestion. "
        "Each prediction is persisted to the database for monitoring and audit."
    ),
    tags=["Prediction"],
)
async def predict(
    body: BookingRequest,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """Predict demand score and suggest an optimal room rate."""
    data = pd.DataFrame([body.model_dump()])
    try:
        result = predict_demand(_xgb_model, _lgbm_model, data, body.room_rate)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction service error") from exc

    request_id = log_prediction(
        db,
        body.model_dump(),
        result["demand_score"],
        result["suggested_rate"],
        settings.model_version,
    )
    return PredictionResponse(
        request_id=request_id,
        demand_score=result["demand_score"],
        demand_tier=result["demand_tier"],
        suggested_rate=result["suggested_rate"],
        model_version=settings.model_version,
    )


@app.get(
    f"{settings.api_prefix}/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns API liveness, model load status, and training AUC.",
    tags=["Operations"],
)
async def health() -> HealthResponse:
    """Return service health and model readiness."""
    return HealthResponse(
        status="healthy",
        model_loaded=_xgb_model is not None,
        model_version=settings.model_version,
        auc_mean=_model_metrics.get("auc_mean"),  # type: ignore[arg-type]
        environment=settings.environment,
        uptime_seconds=round(time.monotonic() - _started_at, 1),
    )


@app.get(
    f"{settings.api_prefix}/metrics",
    summary="Prediction metrics and drift status",
    description=(
        "Returns aggregated prediction statistics and a KS-test drift report "
        "comparing recent production demand scores to the training reference distribution."
    ),
    tags=["Operations"],
)
async def metrics(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return prediction statistics and drift detection results."""
    stats = get_prediction_stats(db)

    recent_preds = db.query(Prediction).order_by(Prediction.timestamp.desc()).limit(50).all()
    recent_scores = [p.demand_score for p in recent_preds]

    drift = compute_drift(_reference_scores, recent_scores)

    return {
        "prediction_stats": stats,
        "drift": drift,
        "model_version": settings.model_version,
        "model_metrics": _model_metrics,
    }


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Redirect-style service banner pointing to interactive docs."""
    return {
        "service": "Suite-Cast",
        "version": settings.model_version,
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
    }


@app.get(
    f"{settings.api_prefix}/model-info",
    summary="Model metadata",
    description="Returns training metrics, feature count, and ensemble composition.",
    tags=["Operations"],
)
async def model_info() -> dict[str, Any]:
    """Return model training metadata and ensemble composition."""
    return {
        "model_version": settings.model_version,
        "ensemble": ["XGBClassifier", "LGBMClassifier"],
        "n_features": _model_metrics.get("n_features"),
        "n_training_samples": _model_metrics.get("n_samples"),
        "cv_folds": 5,
        "metrics": _model_metrics,
    }
