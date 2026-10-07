"""FastAPI application for Sports-Oracle match prediction service."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.database import create_tables
from app.features import make_synthetic_dataset
from app.model import METRICS_PATH, MODEL_PATH, load_model, predict, read_metrics, train_model
from app.monitoring import check_feature_drift, get_recent_predictions, log_prediction

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

_model = None
_request_log: list[dict] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    create_tables()
    if MODEL_PATH.exists():
        try:
            _model = load_model()
            logger.info("model_loaded_at_startup")
        except Exception:
            logger.warning("model_load_failed_at_startup")
    else:
        logger.info("no_model_found_training_on_synthetic_data")
        X, y = make_synthetic_dataset(n=2000)
        _model, _ = train_model(X, y)
        logger.info("synthetic_model_trained")
    yield


app = FastAPI(
    title="Sports-Oracle",
    description="ML API predicting sports match outcomes using XGBoost-LightGBM-RandomForest ensemble.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(GZipMiddleware, minimum_size=500)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Response-Time-Ms"] = str(round(elapsed_ms, 2))
    logger.info(
        "request",
        extra={
            "correlation_id": correlation_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "ms": round(elapsed_ms, 2),
        },
    )
    return response


# ── Pydantic schemas ──────────────────────────────────────────────────────────


class MatchFeatures(BaseModel):
    """Input features for a single match prediction."""

    home_team: str = Field(..., min_length=1, max_length=64, examples=["Arsenal"])
    away_team: str = Field(..., min_length=1, max_length=64, examples=["Chelsea"])
    competition: str = Field(default="Premier League", max_length=64)
    home_form: float = Field(..., ge=0.0, le=1.0, description="Recent win rate (0–1)")
    away_form: float = Field(..., ge=0.0, le=1.0, description="Recent win rate (0–1)")
    home_attack: float = Field(default=1.0, ge=0.0, description="Attack strength index")
    away_attack: float = Field(default=1.0, ge=0.0, description="Attack strength index")
    home_defense: float = Field(default=1.0, ge=0.0, description="Defense strength index")
    away_defense: float = Field(default=1.0, ge=0.0, description="Defense strength index")
    h2h_home_wins: int = Field(default=0, ge=0, description="H2H home team wins")
    h2h_draws: int = Field(default=0, ge=0, description="H2H draws")
    h2h_away_wins: int = Field(default=0, ge=0, description="H2H away team wins")
    home_rest_days: int = Field(default=7, ge=1, le=30, description="Days since last match")
    away_rest_days: int = Field(default=7, ge=1, le=30, description="Days since last match")

    @field_validator("home_team", "away_team")
    @classmethod
    def teams_must_differ(cls, v: str) -> str:
        return v.strip()


class PredictionResponse(BaseModel):
    """Outcome prediction with probabilities."""

    request_id: str
    home_team: str
    away_team: str
    competition: str
    predicted_outcome: str = Field(..., description="H=home win, D=draw, A=away win")
    prob_home: float
    prob_draw: float
    prob_away: float
    confidence: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class MetricsResponse(BaseModel):
    model_version: str
    auc_mean: float
    auc_std: float
    accuracy_mean: float
    n_features: int
    n_samples: int
    recent_predictions: int


class DriftRequest(BaseModel):
    reference_features: list[dict[str, Any]]
    current_features: list[dict[str, Any]]


class BatchRequest(BaseModel):
    matches: list[MatchFeatures] = Field(..., min_length=1, max_length=50)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
@limiter.limit("60/minute")
async def health(request: Request) -> HealthResponse:
    """Return service health status."""
    return HealthResponse(
        status="ok" if _model is not None else "degraded",
        model_loaded=_model is not None,
        version="1.0.0",
    )


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
@limiter.limit("200/minute")
async def predict_match(request: Request, features: MatchFeatures) -> PredictionResponse:
    """Predict the outcome of a football match."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    start = time.perf_counter()

    feat_dict = features.model_dump(exclude={"home_team", "away_team", "competition"})
    X = pd.DataFrame([feat_dict])

    result = predict(_model, X)
    latency_ms = (time.perf_counter() - start) * 1000

    _request_log.append(feat_dict)
    if len(_request_log) > 1000:
        _request_log.pop(0)

    log_prediction(
        home_team=features.home_team,
        away_team=features.away_team,
        competition=features.competition,
        features=feat_dict,
        prediction=result,
        request_id=correlation_id,
        latency_ms=round(latency_ms, 2),
    )

    return PredictionResponse(
        request_id=correlation_id,
        home_team=features.home_team,
        away_team=features.away_team,
        competition=features.competition,
        predicted_outcome=result["predicted_outcome"],
        prob_home=result["prob_home"],
        prob_draw=result["prob_draw"],
        prob_away=result["prob_away"],
        confidence=result["confidence"],
        model_version="1.0.0",
    )


@app.post("/api/v1/predict/batch", tags=["Prediction"])
@limiter.limit("50/minute")
async def predict_batch(request: Request, body: BatchRequest) -> list[dict]:
    """Predict outcomes for multiple matches."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    results = []
    for m in body.matches:
        feat_dict = m.model_dump(exclude={"home_team", "away_team", "competition"})
        X = pd.DataFrame([feat_dict])
        res = predict(_model, X)
        results.append({"home_team": m.home_team, "away_team": m.away_team, **res})
    return results


@app.get("/api/v1/metrics", response_model=MetricsResponse, tags=["Monitoring"])
@limiter.limit("30/minute")
async def metrics(request: Request) -> MetricsResponse:
    """Return model training metrics."""
    m = read_metrics(METRICS_PATH)
    try:
        recent = get_recent_predictions(n=100)
    except Exception:
        recent = []
    return MetricsResponse(
        model_version="1.0.0",
        auc_mean=m.get("auc_mean", 0.0),
        auc_std=m.get("auc_std", 0.0),
        accuracy_mean=m.get("accuracy_mean", 0.0),
        n_features=m.get("n_features", 0),
        n_samples=m.get("n_samples", 0),
        recent_predictions=len(recent),
    )


@app.post("/api/v1/drift", tags=["Monitoring"])
@limiter.limit("20/minute")
async def check_drift(request: Request, body: DriftRequest) -> dict:
    """Run KS-test drift detection across feature columns."""
    results = check_feature_drift(body.reference_features, body.current_features)
    n_drifted = sum(1 for v in results.values() if v.get("drift_detected"))
    return {
        "features_checked": len(results),
        "features_drifted": n_drifted,
        "drift_detected": n_drifted > 0,
        "details": results,
    }


@app.post("/api/v1/retrain", tags=["Model"])
@limiter.limit("5/minute")
async def retrain(request: Request) -> dict:
    """Retrain the model on fresh synthetic data (dev endpoint)."""
    global _model
    X, y = make_synthetic_dataset(n=3000)
    _model, metrics_out = train_model(X, y)
    logger.info("model_retrained_via_api")
    return {"status": "retrained", "metrics": metrics_out}
