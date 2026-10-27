"""FastAPI application with /predict, /health, /metrics, /drift endpoints."""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.model import (
    PredictResult,
    TrainResult,
    generate_synthetic_data,
    load_model,
    predict,
    train,
)
from app.monitoring import get_drift_summary, log_prediction, run_drift_check

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_VERSION: str = os.environ.get("MODEL_VERSION", "1.0.0")

# Global model cache (loaded once at startup)
_pipeline: Any = None
_model: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise the database and warm up the model on startup."""
    global _pipeline, _model
    init_db()
    result = load_model()
    if result is not None:
        _pipeline, _model = result
        logger.info("Model loaded from disk (version %s)", MODEL_VERSION)
    else:
        logger.info("No saved model found — training on synthetic data...")
        samples, labels = generate_synthetic_data(n=500)
        train_result = train(samples, labels, model_version=MODEL_VERSION)
        loaded = load_model()
        if loaded:
            _pipeline, _model = loaded
        logger.info(
            "Bootstrap training complete: AUC=%.4f, F1=%.4f",
            train_result.auc_roc,
            train_result.f1,
        )
    yield


app = FastAPI(
    title="Social-Pulse",
    description=(
        "Viral social media content prediction API. "
        "Predicts whether a post will go viral using an "
        "LightGBM + XGBoost + RandomForest ensemble with NLP features."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: correlation ID and latency logging
# ---------------------------------------------------------------------------


@app.middleware("http")
async def add_correlation_id(request: Request, call_next: Any) -> Response:
    """Attach a unique X-Request-ID header to every response."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
    logger.info(
        "%s %s %s %.1fms [%s]",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request_id,
    )
    return response


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class PostFeatures(BaseModel):
    """Input features for a single social media post."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Raw post text (up to 2000 chars).",
        examples=["Just dropped our new product launch! #innovation #tech @everyone"],
    )
    author_followers: int = Field(..., ge=0, le=500_000_000, description="Author's follower count.")
    has_media: bool = Field(False, description="Post contains an image or video.")
    has_link: bool = Field(False, description="Post contains an external URL.")
    hashtag_count: int = Field(0, ge=0, le=100, description="Number of hashtags.")
    mention_count: int = Field(0, ge=0, le=100, description="Number of @mentions.")
    hour_of_day: int = Field(..., ge=0, le=23, description="UTC hour of posting.")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday).")

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("text must not be blank")
        return stripped


class PredictResponse(BaseModel):
    """Response from the /predict endpoint."""

    request_id: str
    viral_probability: float
    viral_label: bool
    ensemble_confidence: float
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Response from the /metrics endpoint."""

    total_predictions: int
    viral_rate: float
    avg_confidence: float
    avg_latency_ms: float
    model_version: str


class TrainRequest(BaseModel):
    """Request body for the /train endpoint."""

    n_synthetic: int = Field(500, ge=100, le=10000)
    model_version: str = Field("1.0.0", max_length=32)


_startup_time: float = time.time()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
def health_check() -> HealthResponse:
    """Return service health and model status."""
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        model_version=MODEL_VERSION,
        uptime_seconds=time.time() - _startup_time,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
)
def predict_endpoint(
    body: PostFeatures,
    request: Request,
    db: Session = Depends(get_db),
) -> PredictResponse:
    """Predict whether a social post will go viral.

    Returns a probability score (0-1), binary label, and ensemble confidence.
    Every prediction is persisted for monitoring and retraining.
    """
    if _model is None or _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please call /train first.",
        )

    request_id: str = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    sample = body.model_dump()

    try:
        result: PredictResult = predict(sample, _pipeline, _model, MODEL_VERSION)
    except Exception as exc:
        logger.exception("Prediction failed for request %s", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {exc}",
        ) from exc

    try:
        log_prediction(
            db=db,
            request_id=request_id,
            sample=sample,
            viral_probability=result.viral_probability,
            viral_label=result.viral_label,
            ensemble_confidence=result.ensemble_confidence,
            model_version=result.model_version,
            latency_ms=result.latency_ms,
        )
    except Exception:
        logger.warning("Prediction logged but DB write failed for %s", request_id)

    return PredictResponse(
        request_id=request_id,
        viral_probability=result.viral_probability,
        viral_label=result.viral_label,
        ensemble_confidence=result.ensemble_confidence,
        model_version=result.model_version,
        latency_ms=result.latency_ms,
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["Operations"],
)
def metrics_endpoint(db: Session = Depends(get_db)) -> MetricsResponse:
    """Return aggregated prediction metrics from the database."""
    from sqlalchemy import func

    from app.database import Prediction as PredModel

    total = db.query(func.count(PredModel.id)).scalar() or 0
    viral_count = (
        db.query(func.count(PredModel.id)).filter(PredModel.viral_label.is_(True)).scalar() or 0
    )
    avg_conf = db.query(func.avg(PredModel.ensemble_confidence)).scalar() or 0.0
    avg_lat = db.query(func.avg(PredModel.latency_ms)).scalar() or 0.0

    return MetricsResponse(
        total_predictions=total,
        viral_rate=viral_count / total if total else 0.0,
        avg_confidence=float(avg_conf),
        avg_latency_ms=float(avg_lat),
        model_version=MODEL_VERSION,
    )


@app.get("/drift", tags=["Monitoring"])
def drift_endpoint(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return recent KS-test drift detection results."""
    results = get_drift_summary(db, limit=50)
    return {"drift_results": results, "count": len(results)}


@app.post("/drift/check", tags=["Monitoring"])
def run_drift_endpoint(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Trigger a drift check and return the results."""
    records = run_drift_check(db)
    drifted = [r.feature_name for r in records if r.drift_detected]
    return {
        "features_checked": len(records),
        "features_drifted": drifted,
        "drift_detected": len(drifted) > 0,
    }


@app.post("/train", response_model=dict, tags=["Training"])
def train_endpoint(
    body: TrainRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Retrain the model on synthetic data (or real data if available).

    For production use, this endpoint would pull labelled data from the DB.
    """
    global _pipeline, _model
    samples, labels = generate_synthetic_data(n=body.n_synthetic)
    result: TrainResult = train(samples, labels, model_version=body.model_version)
    loaded = load_model()
    if loaded:
        _pipeline, _model = loaded
    return {
        "run_id": result.run_id,
        "model_version": result.model_version,
        "auc_roc": result.auc_roc,
        "f1": result.f1,
        "precision": result.precision,
        "recall": result.recall,
        "cv_auc_mean": sum(result.cv_auc_scores) / len(result.cv_auc_scores)
        if result.cv_auc_scores
        else 0.0,
        "train_samples": result.train_samples,
        "duration_sec": result.duration_sec,
    }


@app.get("/version", tags=["Operations"])
def version_endpoint() -> dict[str, str]:
    """Return the current API and model version."""
    return {"api_version": MODEL_VERSION, "service": "social-pulse"}
