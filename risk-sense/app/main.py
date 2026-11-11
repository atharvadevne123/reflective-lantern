"""FastAPI application for Risk-Sense credit risk scoring."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.database import create_tables, get_db
from app.features import extract_raw_features, risk_category
from app.middleware import CorrelationIdMiddleware, RateLimitMiddleware
from app.model import generate_synthetic_data, load_model, save_model, train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Risk-Sense API",
    description="Credit risk scoring and loan default prediction",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(RateLimitMiddleware)

_model: Any = None
_pipeline: Any = None


@app.on_event("startup")
async def startup_event() -> None:
    global _model, _pipeline
    create_tables()
    try:
        _model, _pipeline = load_model()
        logger.info("Loaded existing model from disk")
    except FileNotFoundError:
        logger.info("No saved model found — training on synthetic data")
        X, y = generate_synthetic_data(n_samples=2000)
        from app.features import build_pipeline
        _pipeline = build_pipeline()
        _model, metrics = train(X, y, _pipeline)
        save_model(_model, _pipeline)
        logger.info("Bootstrap model trained: %s", metrics)


class CreditApplicationRequest(BaseModel):
    applicant_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique applicant identifier")
    debt_to_income: float = Field(..., ge=0.0, le=10.0, description="Debt-to-income ratio (0-10)")
    payment_history_score: float = Field(..., ge=0.0, le=1.0, description="Payment history score (0=poor, 1=perfect)")
    credit_utilization: float = Field(..., ge=0.0, le=1.0, description="Credit utilization ratio (0-1)")
    account_age_months: int = Field(..., ge=1, le=600, description="Average age of credit accounts in months")
    num_open_accounts: int = Field(..., ge=0, le=50, description="Number of open credit accounts")
    num_derogatory_marks: int = Field(..., ge=0, le=20, description="Number of derogatory marks")
    annual_income: float = Field(..., ge=0.0, description="Annual income in USD")

    @field_validator("debt_to_income")
    @classmethod
    def validate_dti(cls, v: float) -> float:
        if v < 0:
            raise ValueError("debt_to_income must be non-negative")
        return v


class CreditScoreResponse(BaseModel):
    applicant_id: str
    risk_score: float = Field(..., description="Risk score (0=low risk, 1=high risk)")
    risk_category: str = Field(..., description="Risk category: low, medium, high, very_high")
    default_probability: float = Field(..., description="Probability of default (0-1)")
    recommendation: str = Field(..., description="Lending recommendation")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class MetricsResponse(BaseModel):
    total_predictions: int
    avg_default_probability: float
    high_risk_rate: float


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        version="1.0.0",
    )


@app.get("/api/v1/version", tags=["System"])
async def version() -> dict[str, str]:
    """Return API version."""
    return {"version": "1.0.0", "model": "xgb+lgbm+rf-ensemble"}


@app.post("/api/v1/score", response_model=CreditScoreResponse, tags=["Scoring"])
async def score_application(
    request: CreditApplicationRequest,
    db: Session = Depends(get_db),
) -> CreditScoreResponse:
    """Score a credit application and return risk assessment."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features_dict = request.model_dump()
        X_raw = extract_raw_features(features_dict)
        X_transformed = _pipeline.transform(X_raw) if _pipeline else X_raw
        default_prob = float(_model.predict_proba(X_transformed)[0, 1])

        cat = risk_category(default_prob)
        recommendation = {
            "low": "Approve — low risk applicant",
            "medium": "Approve with standard terms",
            "high": "Review manually — elevated risk",
            "very_high": "Decline — high default probability",
        }.get(cat, "Review manually")

        try:
            from app.monitoring import log_prediction
            log_prediction(
                db_session=db,
                applicant_id=request.applicant_id,
                risk_score=default_prob,
                risk_category=cat,
                default_probability=default_prob,
                features=features_dict,
            )
        except Exception as e:
            logger.warning("Failed to log prediction: %s", e)

        return CreditScoreResponse(
            applicant_id=request.applicant_id,
            risk_score=round(default_prob, 4),
            risk_category=cat,
            default_probability=round(default_prob, 4),
            recommendation=recommendation,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.error("Scoring error: %s", e)
        raise HTTPException(status_code=500, detail="Internal scoring error") from e


@app.get("/api/v1/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def metrics(db: Session = Depends(get_db)) -> MetricsResponse:
    """Return aggregate prediction metrics."""
    from app.database import CreditApplication
    try:
        records = db.query(CreditApplication).all()
        if not records:
            return MetricsResponse(total_predictions=0, avg_default_probability=0.0, high_risk_rate=0.0)
        probs = [r.default_probability for r in records if r.default_probability is not None]
        high_risk = sum(1 for p in probs if p >= 0.5)
        return MetricsResponse(
            total_predictions=len(records),
            avg_default_probability=round(float(np.mean(probs)), 4) if probs else 0.0,
            high_risk_rate=round(high_risk / len(probs), 4) if probs else 0.0,
        )
    except Exception as e:
        logger.error("Metrics error: %s", e)
        raise HTTPException(status_code=500, detail="Metrics unavailable") from e


@app.get("/api/v1/drift", tags=["Monitoring"])
async def drift_report(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return drift report comparing recent vs historical predictions."""
    from app.database import CreditApplication
    from app.monitoring import ks_drift_test

    records = db.query(CreditApplication).order_by(CreditApplication.created_at.asc()).all()
    if len(records) < 20:
        return {"status": "insufficient_data", "min_required": 20, "current": len(records)}

    mid = len(records) // 2
    reference = [r.default_probability for r in records[:mid] if r.default_probability is not None]
    current = [r.default_probability for r in records[mid:] if r.default_probability is not None]

    return {"drift_analysis": ks_drift_test(reference, current), "reference_n": len(reference), "current_n": len(current)}
