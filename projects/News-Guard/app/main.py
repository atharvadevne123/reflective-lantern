"""News-Guard FastAPI application."""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db, init_db, log_prediction
from app.features import build_feature_dataframe, compute_feature_hash
from app.model import NewsGuardModel
from app.monitoring import (
    compute_drift,
    get_monitoring_summary,
    get_recent_predictions,
    log_drift_check,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s [%(correlation_id)s] %(message)s",
    defaults={"correlation_id": "-"},
)
logger = logging.getLogger(__name__)

_model: NewsGuardModel | None = None
_reference_scores: list[float] = []

APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _reference_scores
    init_db()
    _model = NewsGuardModel.load()
    _reference_scores = _model.get_reference_features()
    logger.info("News-Guard ready. Model version=%s", _model.version)
    yield
    logger.info("News-Guard shutting down.")


app = FastAPI(
    title="News-Guard",
    description="Real-time fake news and misinformation classification API using LightGBM-XGBoost-RF ensemble.",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    cid = request.headers.get("X-Correlation-ID", str(uuid.uuid4())[:8])
    request.state.correlation_id = cid
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Correlation-ID"] = cid
    response.headers["X-Response-Time-Ms"] = str(elapsed)
    return response


class ArticleRequest(BaseModel):
    headline: str = Field(..., min_length=5, max_length=500, description="Article headline")
    body: str = Field(default="", max_length=10000, description="Article body text")
    source_domain: str = Field(default="unknown", max_length=255, description="Publisher domain")
    publication_hour: int = Field(default=12, ge=0, le=23)
    share_count: int = Field(default=0, ge=0)


class PredictionResponse(BaseModel):
    article_id: str
    prediction: str
    confidence: float
    credibility_score: float
    fake_probability: float
    real_probability: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool


class MetricsResponse(BaseModel):
    total_predictions: int
    fake_count: int
    real_count: int
    fake_rate: float
    drift_detected: bool
    last_drift_check: str | None


@app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Service liveness check."""
    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        model_loaded=_model is not None,
    )


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(
    req: ArticleRequest,
    db: Session = Depends(get_db),
):
    """Classify an article as FAKE or REAL with confidence and credibility scores."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    article_id = str(uuid.uuid4())[:12]
    df = build_feature_dataframe(
        headline=req.headline,
        body=req.body,
        source_domain=req.source_domain,
        publication_hour=req.publication_hour,
        share_count=req.share_count,
    )
    result = _model.predict(df)
    feature_hash = compute_feature_hash({
        "headline": req.headline,
        "source_domain": req.source_domain,
    })

    log_prediction(
        db=db,
        article_id=article_id,
        headline=req.headline[:200],
        source_domain=req.source_domain,
        prediction=result["prediction"],
        confidence=result["confidence"],
        credibility_score=result["credibility_score"],
        feature_hash=feature_hash,
        model_version=_model.version,
    )

    logger.info(
        "Predict article_id=%s prediction=%s confidence=%.4f",
        article_id,
        result["prediction"],
        result["confidence"],
    )

    return PredictionResponse(
        article_id=article_id,
        model_version=_model.version,
        **result,
    )


@app.get("/api/v1/metrics", response_model=MetricsResponse, tags=["monitoring"])
async def metrics(db: Session = Depends(get_db)):
    """Return prediction statistics and drift status."""
    summary = get_monitoring_summary(db)
    recent = get_recent_predictions(db, hours=24)

    if recent and _reference_scores:
        drift_result = compute_drift(_reference_scores, recent)
        log_drift_check(
            db=db,
            feature_name="fake_probability",
            ks_statistic=drift_result["ks_statistic"],
            p_value=drift_result["p_value"],
            drift_detected=drift_result["drift_detected"],
        )
        summary["drift_detected"] = drift_result["drift_detected"]

    return MetricsResponse(**summary)


@app.get("/api/v1/drift", tags=["monitoring"])
async def drift_check(db: Session = Depends(get_db)):
    """Run an on-demand KS-drift check against the reference distribution."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    recent = get_recent_predictions(db, hours=24)
    result = compute_drift(_reference_scores, recent)
    return {"reference_size": len(_reference_scores), "current_size": len(recent), **result}
