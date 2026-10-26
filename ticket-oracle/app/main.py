"""FastAPI application for Ticket-Oracle.

Endpoints:
  GET  /api/v1/health          — liveness + model status
  POST /api/v1/predict         — single ticket priority & SLA prediction
  POST /api/v1/predict/batch   — batch predictions (up to 50 tickets)
  GET  /api/v1/metrics         — model performance metrics
  GET  /api/v1/drift           — recent drift check results
  POST /api/v1/train           — trigger model retraining on synthetic data
  GET  /api/v1/predictions     — recent prediction history (last 100)
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.features import make_synthetic_dataset, payload_to_dataframe
from app.model import METRICS_PATH, load_models, predict, train_models
from app.monitoring import get_recent_predictions, log_prediction, run_drift_check

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","message":"%(message)s"}',
)
logger = logging.getLogger(__name__)

MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

limiter = Limiter(key_func=get_remote_address)

_priority_model = None
_sla_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise DB and attempt to load pre-trained models on startup."""
    global _priority_model, _sla_model
    init_db()
    try:
        _priority_model, _sla_model = load_models()
        logger.info("startup_models_loaded")
    except FileNotFoundError:
        logger.warning("startup_models_not_found_training_now")
        X, y_p, y_s = make_synthetic_dataset()
        train_models(X, y_p, y_s)
        _priority_model, _sla_model = load_models()
    yield
    logger.info("shutdown_complete")


app = FastAPI(
    title="Ticket-Oracle",
    description=(
        "IT helpdesk ticket priority classification and SLA breach prediction API "
        "using XGBoost-LightGBM-RandomForest ensemble with NLP text features, "
        "workload lag metrics, and automated drift monitoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Attach a correlation ID to every request/response cycle."""
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    start = time.monotonic()
    response: Response = await call_next(request)
    elapsed = round((time.monotonic() - start) * 1000, 2)
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Response-Time-Ms"] = str(elapsed)
    logger.info(
        "request_handled",
        extra={
            "correlation_id": correlation_id,
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "elapsed_ms": elapsed,
        },
    )
    return response


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class TicketPayload(BaseModel):
    """Input schema for a single ticket prediction request."""

    ticket_id: str = Field(..., min_length=1, max_length=64, description="Unique ticket identifier")
    description: str = Field(..., min_length=3, max_length=2000, description="Ticket description text")
    department: str = Field("Unknown", max_length=64, description="Reporting department")
    incident_type: str = Field("Unknown", max_length=64, description="Incident classification")
    channel: str = Field("email", description="Submission channel: email|phone|chat|portal")
    customer_tier: str = Field("Standard", description="Customer tier: Bronze|Standard|Silver|Gold")
    product_area: str = Field("Unknown", max_length=64, description="Affected product or service area")
    open_tickets_count: int = Field(10, ge=0, le=10000, description="Current open ticket count in queue")
    agent_load: int = Field(5, ge=0, le=100, description="Number of tickets assigned to current agent")
    hour_of_day: int = Field(12, ge=0, le=23, description="Hour when ticket was submitted (0-23)")
    day_of_week: int = Field(0, ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    day_of_month: int = Field(15, ge=1, le=31, description="Day of month for month-end detection")
    ticket_age_minutes: float = Field(0.0, ge=0.0, description="Minutes since ticket was opened")
    same_user_last_7d: int = Field(0, ge=0, description="Tickets submitted by same user in last 7 days")

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v: str) -> str:
        allowed = {"email", "phone", "chat", "portal"}
        if v.lower() not in allowed:
            raise ValueError(f"channel must be one of {allowed}")
        return v.lower()

    @field_validator("customer_tier")
    @classmethod
    def validate_tier(cls, v: str) -> str:
        allowed = {"Bronze", "Standard", "Silver", "Gold"}
        if v.title() not in allowed:
            raise ValueError(f"customer_tier must be one of {allowed}")
        return v.title()


class BatchTicketPayload(BaseModel):
    """Batch prediction request (up to 50 tickets)."""

    tickets: list[TicketPayload] = Field(..., min_length=1, max_length=50)


class PredictionResponse(BaseModel):
    """Output schema for a single ticket prediction."""

    ticket_id: str
    priority: str
    priority_probabilities: dict[str, float]
    sla_breach_risk: float
    sla_breach_predicted: bool
    estimated_resolution_hours: float
    confidence: float
    model_version: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/v1/health", tags=["Operations"])
@limiter.limit("60/minute")
async def health(request: Request, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return API liveness status and model availability.

    Checks whether both ML models are loaded and the database is reachable.
    """
    model_loaded = _priority_model is not None and _sla_model is not None
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    status = "healthy" if model_loaded and db_ok else "degraded"
    return {"status": status, "model_loaded": model_loaded, "db_ok": db_ok, "version": MODEL_VERSION}


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
@limiter.limit("200/minute")
async def predict_priority(
    request: Request,
    payload: TicketPayload,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """Predict ticket priority (P1-P4) and SLA breach risk.

    Returns priority class, per-class probabilities, SLA breach probability,
    estimated resolution time, and prediction confidence.
    """
    if _priority_model is None or _sla_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    X = payload_to_dataframe(payload.model_dump())
    result = predict(_priority_model, _sla_model, X)

    log_prediction(db, payload.ticket_id, result, MODEL_VERSION)

    return PredictionResponse(ticket_id=payload.ticket_id, model_version=MODEL_VERSION, **result)


@app.post("/api/v1/predict/batch", tags=["Prediction"])
@limiter.limit("50/minute")
async def predict_batch(
    request: Request,
    payload: BatchTicketPayload,
    db: Session = Depends(get_db),
) -> list[PredictionResponse]:
    """Predict priority and SLA breach risk for up to 50 tickets at once."""
    if _priority_model is None or _sla_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    responses = []
    for ticket in payload.tickets:
        X = payload_to_dataframe(ticket.model_dump())
        result = predict(_priority_model, _sla_model, X)
        log_prediction(db, ticket.ticket_id, result, MODEL_VERSION)
        responses.append(PredictionResponse(ticket_id=ticket.ticket_id, model_version=MODEL_VERSION, **result))
    return responses


@app.get("/api/v1/metrics", tags=["Operations"])
@limiter.limit("30/minute")
async def metrics(request: Request) -> dict[str, Any]:
    """Return current model performance metrics from the last training run."""
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="No metrics available; model not yet trained")
    with open(METRICS_PATH) as f:
        return json.load(f)


@app.get("/api/v1/drift", tags=["Operations"])
@limiter.limit("30/minute")
async def drift_check(
    request: Request,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Run KS-test drift detection on recent prediction probabilities.

    Compares the last 500 predictions against a synthetic reference
    distribution to detect model drift.
    """
    recent = get_recent_predictions(db, n=500)
    if len(recent) < 20:
        return {"status": "insufficient_data", "n_predictions": len(recent), "drift_results": []}

    reference_sla = [0.3 + 0.2 * i / 500 for i in range(500)]
    current_sla = [float(r.sla_breach_risk) for r in recent]

    drift_result = run_drift_check(db, "sla_breach_risk", reference_sla, current_sla)
    return {
        "status": "ok",
        "n_predictions": len(recent),
        "drift_results": [drift_result],
    }


@app.post("/api/v1/train", tags=["Operations"])
@limiter.limit("5/minute")
async def trigger_training(request: Request) -> dict[str, Any]:
    """Trigger model retraining on fresh synthetic data.

    Generates 2000 synthetic tickets, trains both models with 5-fold CV,
    and replaces the serving models in memory.
    """
    global _priority_model, _sla_model
    X, y_p, y_s = make_synthetic_dataset(n_samples=2000)
    metrics_out = train_models(X, y_p, y_s, model_version=MODEL_VERSION)
    _priority_model, _sla_model = load_models()
    logger.info("models_retrained_via_api")
    return {"status": "retrained", "metrics": metrics_out}


@app.get("/api/v1/predictions", tags=["Operations"])
@limiter.limit("30/minute")
async def recent_predictions(
    request: Request,
    db: Session = Depends(get_db),
    n: int = 100,
) -> list[dict[str, Any]]:
    """Return the n most recent predictions (default 100, max 500)."""
    n = min(max(n, 1), 500)
    rows = get_recent_predictions(db, n=n)
    return [
        {
            "id": r.id,
            "ticket_id": r.ticket_id,
            "priority_predicted": r.priority_predicted,
            "sla_breach_risk": r.sla_breach_risk,
            "confidence": r.confidence,
            "model_version": r.model_version,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]
