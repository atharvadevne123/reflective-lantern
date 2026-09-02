"""FastAPI application for Cyber-Guard intrusion detection API."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.anomaly import ensure_anomaly_model_exists, load_anomaly_detector, score_anomaly
from app.config import get_settings
from app.database import create_tables, get_db
from app.features import make_sample_df
from app.model import ensure_model_exists, load_model, predict
from app.monitoring import get_prediction_stats, log_prediction, run_drift_check
from app.rate_limit import RateLimitMiddleware

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

_pipeline = None
_label_encoder = None
_anomaly_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and create tables on startup; log on shutdown."""
    global _pipeline, _label_encoder, _anomaly_pipeline
    create_tables()
    ensure_model_exists()
    ensure_anomaly_model_exists()
    _pipeline, _label_encoder = load_model()
    _anomaly_pipeline = load_anomaly_detector()
    logger.info("models loaded version=%s", settings.api_version)
    yield
    logger.info("shutting down")


app = FastAPI(
    title="Cyber-Guard",
    description="Real-time network intrusion detection and threat classification API.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware, limit_per_minute=settings.rate_limit_per_minute)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    start = time.time()
    response: Response = await call_next(request)
    elapsed = round((time.time() - start) * 1000, 2)
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Response-Time-Ms"] = str(elapsed)
    logger.info("request method=%s path=%s status=%d ms=%s cid=%s",
                request.method, request.url.path, response.status_code, elapsed, correlation_id)
    return response


# --- Pydantic schemas ---

class NetworkConnectionRequest(BaseModel):
    src_bytes: float = Field(..., ge=0, description="Source bytes transferred")
    dst_bytes: float = Field(..., ge=0, description="Destination bytes transferred")
    duration: float = Field(..., ge=0, description="Connection duration in seconds")
    protocol_type: str = Field(..., description="Protocol type: tcp, udp, icmp")
    service: str = Field(..., description="Network service: http, ftp, ssh, etc.")
    flag: str = Field(..., description="Connection flag: SF, S0, REJ, etc.")

    model_config = {"json_schema_extra": {"example": {
        "src_bytes": 491, "dst_bytes": 0, "duration": 0,
        "protocol_type": "tcp", "service": "http", "flag": "SF",
    }}}


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    class_probabilities: dict[str, float]
    correlation_id: str | None = None
    anomaly: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Unsupervised outlier assessment. When is_anomaly is true the "
            "connection sits outside the training distribution, so the "
            "supervised label above should be treated with low trust."
        ),
    )


class AnomalyResponse(BaseModel):
    anomaly_score: float
    decision_score: float
    is_anomaly: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class MetricsResponse(BaseModel):
    total_predictions: int
    hours: int
    avg_confidence: float | None
    class_counts: dict[str, int]
    drift_check: dict[str, Any] | None = None


# --- API routes ---

@app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=_pipeline is not None,
        version="1.0.0",
    )


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict_intrusion(
    request: Request,
    payload: NetworkConnectionRequest,
    db: Session = Depends(get_db),
):
    """Classify a network connection as normal or intrusion threat category."""
    if _pipeline is None or _label_encoder is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    if payload.protocol_type not in ["tcp", "udp", "icmp"]:
        raise HTTPException(status_code=422, detail=f"unknown protocol_type: {payload.protocol_type}")

    features = payload.model_dump()
    df = make_sample_df(**features)

    result = predict(df, _pipeline, _label_encoder)

    anomaly = score_anomaly(df, _anomaly_pipeline) if _anomaly_pipeline is not None else None

    cid = request.headers.get("X-Correlation-ID")
    log_prediction(db, features, result["prediction"], result["confidence"])

    return PredictionResponse(**result, correlation_id=cid, anomaly=anomaly)


@app.post("/api/v1/anomaly", response_model=AnomalyResponse, tags=["prediction"])
async def anomaly_check(payload: NetworkConnectionRequest):
    """Score a connection for outlier-ness without assigning a threat class.

    Useful for triaging traffic the supervised model has no class for --
    a novel attack looks like an inlier to a classifier forced to pick one
    of five labels, but like an outlier here.
    """
    if _anomaly_pipeline is None:
        raise HTTPException(status_code=503, detail="anomaly model not loaded")
    df = make_sample_df(**payload.model_dump())
    return AnomalyResponse(**score_anomaly(df, _anomaly_pipeline))


@app.get("/api/v1/metrics", response_model=MetricsResponse, tags=["monitoring"])
async def metrics(hours: int = 24, run_drift: bool = False, db: Session = Depends(get_db)):
    """Return prediction statistics and optional drift check for the last N hours."""
    stats = get_prediction_stats(db, hours=hours)
    drift = run_drift_check(db) if run_drift else None
    return MetricsResponse(
        total_predictions=stats["total"],
        hours=stats["hours"],
        avg_confidence=stats.get("avg_confidence"),
        class_counts=stats.get("class_counts", {}),
        drift_check=drift,
    )


@app.get("/api/v1/drift", tags=["monitoring"])
async def drift_check(db: Session = Depends(get_db)):
    """Run KS-test drift check on src_bytes distribution."""
    return run_drift_check(db)
