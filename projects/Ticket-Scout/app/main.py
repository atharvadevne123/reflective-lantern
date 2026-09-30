"""FastAPI application for Ticket-Scout: support ticket triage and SLA prediction."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.retriever import TicketRetriever
from app.database import create_tables, get_db
from app.model import MODEL_DIR, MODEL_VERSION, load_models, predict, train_models
from app.rate_limit import RateLimitMiddleware
from app.monitoring import (
    build_current_window,
    check_all_drift,
    get_recent_predictions,
    log_prediction,
    set_reference_distribution,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

_models: dict[str, Any] = {}
_startup_time: float = 0.0
_retriever: TicketRetriever = TicketRetriever(dim=218)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_time
    _startup_time = time.time()
    logger.info("Ticket-Scout starting up")
    create_tables()
    cat_pipe, breach_pipe, res_pipe = load_models()
    _models["category"] = cat_pipe
    _models["breach"] = breach_pipe
    _models["resolution"] = res_pipe

    # Seed reference distributions from training stats
    from app.model import generate_synthetic_data
    df = generate_synthetic_data(n_samples=500)
    set_reference_distribution("text_length", (df["subject"].str.len() + df["body"].str.len()).tolist())
    set_reference_distribution("sla_breach_prob", df["sla_breach"].astype(float).tolist())
    set_reference_distribution("resolution_hours_pred", df["resolution_hours"].tolist())
    set_reference_distribution("confidence", [0.75] * 500)
    logger.info("Models and reference distributions loaded")
    yield
    logger.info("Ticket-Scout shutting down")


app = FastAPI(
    title="Ticket-Scout",
    description=(
        "Intelligent IT support ticket triage API. "
        "Auto-classifies issue type, predicts SLA breach risk, "
        "and estimates resolution time using LightGBM ensemble with NLP features."
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


app.add_middleware(RateLimitMiddleware, requests_per_minute=60)

@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Attach a correlation ID to every request and response."""
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    request.state.correlation_id = correlation_id
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Response-Time-Ms"] = f"{elapsed:.1f}"
    logger.info(
        "method=%s path=%s status=%d ms=%.1f cid=%s",
        request.method, request.url.path, response.status_code, elapsed, correlation_id,
    )
    return response


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class TicketRequest(BaseModel):
    """A single support ticket to triage."""

    ticket_id: str | None = Field(default=None, description="Optional caller-supplied ticket ID")
    subject: str = Field(..., min_length=3, max_length=500, description="Ticket subject line")
    body: str = Field(default="", max_length=5000, description="Ticket body text")
    priority: str = Field(default="medium", description="Reported priority: low/medium/high/critical")
    created_hour: int = Field(default=9, ge=0, le=23)
    created_dow: int = Field(default=0, ge=0, le=6, description="Day of week (0=Mon)")
    org_size: int = Field(default=500, ge=1, le=100000)

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        allowed = {"low", "medium", "high", "critical"}
        v = v.lower()
        if v not in allowed:
            raise ValueError(f"priority must be one of {allowed}")
        return v


class BatchRequest(BaseModel):
    """Batch of up to 50 tickets."""

    tickets: list[TicketRequest] = Field(..., min_length=1, max_length=50)


class TicketPrediction(BaseModel):
    """Prediction result for a single ticket."""

    ticket_id: str
    predicted_category: str
    confidence: float
    sla_breach_prob: float
    resolution_hours_pred: float
    risk_level: str


class BatchPredictionResponse(BaseModel):
    predictions: list[TicketPrediction]
    model_version: str


def _risk_level(breach_prob: float) -> str:
    if breach_prob >= 0.6:
        return "high"
    if breach_prob >= 0.3:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/health", tags=["ops"], summary="Check service health", response_description="Health status and uptime")
def health() -> dict[str, Any]:
    """Return service health status and uptime."""
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - _startup_time, 1),
        "model_version": MODEL_VERSION,
        "models_loaded": len(_models) == 3,
    }


@app.post("/api/v1/predict", response_model=TicketPrediction, tags=["inference"])
def predict_single(
    req: TicketRequest,
    db: Session = Depends(get_db),
) -> TicketPrediction:
    """Triage a single support ticket.

    Returns predicted category, SLA breach probability, and resolution time.
    """
    if not _models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    ticket_id = req.ticket_id or str(uuid.uuid4())
    df = pd.DataFrame([req.model_dump()])

    try:
        results = predict(df, _models["category"], _models["breach"], _models["resolution"])
    except Exception as exc:
        logger.exception("Prediction failed for ticket_id=%s", ticket_id)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    r = results[0]
    log_prediction(
        db=db,
        ticket_id=ticket_id,
        subject=req.subject,
        body=req.body,
        priority=req.priority,
        predicted_category=r["predicted_category"],
        sla_breach_prob=r["sla_breach_prob"],
        resolution_hours_pred=r["resolution_hours_pred"],
        confidence=r["confidence"],
        model_version=MODEL_VERSION,
    )

    return TicketPrediction(
        ticket_id=ticket_id,
        predicted_category=r["predicted_category"],
        confidence=round(r["confidence"], 4),
        sla_breach_prob=round(r["sla_breach_prob"], 4),
        resolution_hours_pred=round(r["resolution_hours_pred"], 2),
        risk_level=_risk_level(r["sla_breach_prob"]),
    )


@app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
def predict_batch(
    req: BatchRequest,
    db: Session = Depends(get_db),
) -> BatchPredictionResponse:
    """Triage a batch of up to 50 tickets in one call."""
    if not _models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    records = [t.model_dump() for t in req.tickets]
    df = pd.DataFrame(records)
    ticket_ids = [r.get("ticket_id") or str(uuid.uuid4()) for r in records]
    df["ticket_id"] = ticket_ids

    try:
        results = predict(df, _models["category"], _models["breach"], _models["resolution"])
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {exc}") from exc

    predictions = []
    for i, r in enumerate(results):
        log_prediction(
            db=db,
            ticket_id=ticket_ids[i],
            subject=req.tickets[i].subject,
            body=req.tickets[i].body,
            priority=req.tickets[i].priority,
            predicted_category=r["predicted_category"],
            sla_breach_prob=r["sla_breach_prob"],
            resolution_hours_pred=r["resolution_hours_pred"],
            confidence=r["confidence"],
            model_version=MODEL_VERSION,
        )
        predictions.append(TicketPrediction(
            ticket_id=ticket_ids[i],
            predicted_category=r["predicted_category"],
            confidence=round(r["confidence"], 4),
            sla_breach_prob=round(r["sla_breach_prob"], 4),
            resolution_hours_pred=round(r["resolution_hours_pred"], 2),
            risk_level=_risk_level(r["sla_breach_prob"]),
        ))

    return BatchPredictionResponse(predictions=predictions, model_version=MODEL_VERSION)


@app.get("/api/v1/metrics", tags=["ops"])
def metrics(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return model metrics, prediction volume, and drift summary."""
    metrics_path = MODEL_DIR / "metrics.json"
    model_metrics: dict[str, Any] = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            model_metrics = json.load(f)

    recent = get_recent_predictions(db, hours=24)
    breach_count = sum(1 for p in recent if p.sla_breach_prob >= 0.5)

    return {
        "model_metrics": model_metrics,
        "prediction_volume_24h": len(recent),
        "predicted_breach_count_24h": breach_count,
        "drift_available": True,
    }


@app.post("/api/v1/drift", tags=["monitoring"])
def run_drift_check(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Trigger a drift check on the last 24 hours of predictions."""
    recent = get_recent_predictions(db, hours=24)
    if len(recent) < 5:
        return {"message": "Insufficient predictions for drift check", "results": []}

    window = build_current_window(recent)
    results = check_all_drift(window, db)
    drift_count = sum(1 for r in results if r["drift_detected"])
    return {
        "features_checked": len(results),
        "drift_detected_count": drift_count,
        "results": results,
    }


@app.post("/api/v1/retrain", tags=["ops"])
def retrain() -> dict[str, Any]:
    """Trigger model retraining on fresh synthetic data (demo endpoint)."""
    logger.info("Manual retrain triggered")
    metrics_result = train_models()
    cat_pipe, breach_pipe, res_pipe = load_models()
    _models["category"] = cat_pipe
    _models["breach"] = breach_pipe
    _models["resolution"] = res_pipe
    logger.info("Retrain complete")
    return {"status": "retrained", "metrics": metrics_result}


@app.get("/api/v1/similar/{ticket_id}", tags=["inference"])
def similar_tickets(ticket_id: str) -> dict:
    """Return similar tickets from the in-memory index (demo)."""
    return {
        "ticket_id": ticket_id,
        "similar_count": _retriever.size,
        "message": "Retriever index contains in-session predictions.",
    }
