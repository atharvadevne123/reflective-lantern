"""FastAPI application for Support-Sage ticket triage service."""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.batch import router as batch_router
from app.database import get_db, init_db
from app.model import load_model, predict, train_model
from app.monitoring import get_drift_report, log_prediction
from app.rag_endpoint import router as rag_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

_model_cache: dict[str, Any] = {}


def get_model() -> tuple:
    if "model" not in _model_cache:
        try:
            pipe, le, reg = load_model()
            _model_cache["model"] = pipe
            _model_cache["encoder"] = le
            _model_cache["regressor"] = reg
        except FileNotFoundError:
            logger.warning("Model not found — training on synthetic data")
            from app.features import generate_synthetic_data

            df = generate_synthetic_data(2000)
            pipe, _ = train_model(df)
            pipe, le, reg = load_model()
            _model_cache["model"] = pipe
            _model_cache["encoder"] = le
            _model_cache["regressor"] = reg
    return _model_cache["model"], _model_cache["encoder"], _model_cache["regressor"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Support-Sage starting up — initialising database")
    init_db()
    get_model()
    logger.info("Support-Sage ready")
    yield
    logger.info("Support-Sage shutting down")


app = FastAPI(
    title="Support-Sage",
    description="IT helpdesk ticket triage API: priority classification, category detection, resolution time estimation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(batch_router)
app.include_router(rag_router)


@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = round((time.monotonic() - start) * 1000, 2)
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Response-Time-Ms"] = str(duration_ms)
    logger.info(
        "req=%s path=%s status=%s duration_ms=%s", correlation_id, request.url.path, response.status_code, duration_ms
    )
    return response


# -- Request / Response schemas -----------------------------------------------


class TicketRequest(BaseModel):
    ticket_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ticket identifier")
    subject: str = Field(..., min_length=3, max_length=512, description="Ticket subject line")
    description: str = Field(..., min_length=10, max_length=4096, description="Full ticket description")


class PriorityResponse(BaseModel):
    ticket_id: str
    priority: str
    confidence: float
    priority_probabilities: dict[str, float]
    resolution_hours_estimate: float
    model_version: str = "1.0.0"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


# -- Endpoints -----------------------------------------------------------------


@app.post("/api/v1/predict", response_model=PriorityResponse, summary="Triage a helpdesk ticket")
async def predict_endpoint(
    ticket: TicketRequest,
    db: Session = Depends(get_db),
) -> PriorityResponse:
    """
    Predict priority, category, and estimated resolution time for a helpdesk ticket.

    Returns priority (critical/high/medium/low), confidence score, and resolution hours estimate.
    """
    try:
        model, encoder, regressor = get_model()
        result = predict(ticket.subject, ticket.description, model, encoder, regressor)
        result["category"] = "general"  # category from FAISS RAG in /rag endpoint
        log_prediction(db, ticket.ticket_id, ticket.subject, ticket.description, result)
        return PriorityResponse(ticket_id=ticket.ticket_id, **result)
    except Exception as exc:
        logger.exception("Prediction failed for ticket %s", ticket.ticket_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/v1/health", response_model=HealthResponse, summary="Service health check")
async def health() -> HealthResponse:
    """Return service health status and model availability."""
    model_loaded = "model" in _model_cache
    return HealthResponse(status="ok", model_loaded=model_loaded, version="1.0.0")


@app.get("/api/v1/metrics", summary="Model metrics and drift report")
async def metrics_endpoint(db: Session = Depends(get_db)) -> dict[str, Any]:
    """
    Return model training metrics and feature drift statistics.

    Includes KS-test results for confidence scores, resolution times, and word count distributions.
    """
    import json
    from pathlib import Path

    model_metrics: dict = {}
    metrics_path = Path("models/metrics.json")
    if metrics_path.exists():
        model_metrics = json.loads(metrics_path.read_text())
    drift_report = get_drift_report(db)
    return {"model_metrics": model_metrics, "drift_report": drift_report}


@app.post("/api/v1/train", summary="Trigger model retraining on synthetic data")
async def train_endpoint() -> dict[str, Any]:
    """Retrain the model on fresh synthetic data and reload the in-memory cache."""
    from app.features import generate_synthetic_data

    df = generate_synthetic_data(3000)
    _, metrics = train_model(df)
    _model_cache.clear()
    get_model()
    return {"status": "retrained", "metrics": metrics}
