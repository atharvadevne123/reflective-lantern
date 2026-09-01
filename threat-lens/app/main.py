"""Threat-Lens FastAPI application — network intrusion detection API."""

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db, init_db
from app.features import NetworkFeatureEngineer
from app.logging_config import configure_logging
from app.middleware import RateLimiter, build_rate_limit_middleware
from app.model import load_metrics, load_model, predict
from app.monitoring import get_drift_summary, log_prediction, run_full_drift_check
from app.rag_retriever import ThreatIntelRetriever

settings = get_settings()
configure_logging(level=settings.log_level)
logger = logging.getLogger(__name__)

rate_limiter = RateLimiter(limit_per_minute=settings.rate_limit_per_minute)

_model = None
_retriever: ThreatIntelRetriever | None = None
_reference_data: dict[str, list[float]] = {
    "src_bytes": [],
    "dst_bytes": [],
    "duration": [],
    "confidence": [],
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _model, _retriever
    init_db()
    _model = load_model()
    _retriever = ThreatIntelRetriever()
    _retriever.build_index()
    logger.info("Threat-Lens startup complete")
    yield
    logger.info("Threat-Lens shutdown")


app = FastAPI(
    title="Threat-Lens",
    description="Network intrusion detection API with XGBoost/LightGBM/RF ensemble",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(build_rate_limit_middleware(rate_limiter))


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next: Any) -> Any:
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    request.state.correlation_id = correlation_id
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Response-Time-Ms"] = f"{elapsed:.2f}"
    return response


# ──────────────────────────── Request / Response schemas ──────────────────────


class NetworkFlow(BaseModel):
    """Single network flow observation for classification."""

    duration: float = Field(default=0.0, ge=0, description="Connection duration in seconds")
    src_bytes: float = Field(default=0.0, ge=0, description="Bytes sent from source")
    dst_bytes: float = Field(default=0.0, ge=0, description="Bytes sent to destination")
    land: int = Field(default=0, ge=0, le=1, description="1 if src/dst host:port are identical")
    wrong_fragment: int = Field(default=0, ge=0, description="Number of wrong fragments")
    urgent: int = Field(default=0, ge=0, description="Number of urgent packets")
    hot: int = Field(default=0, ge=0, description="Number of hot indicators")
    num_failed_logins: int = Field(default=0, ge=0, description="Number of failed login attempts")
    logged_in: int = Field(default=0, ge=0, le=1, description="1 if successfully logged in")
    num_compromised: int = Field(default=0, ge=0, description="Number of compromised conditions")
    count: int = Field(default=1, ge=1, description="Connections to same host in past 2 seconds")
    srv_count: int = Field(
        default=1, ge=1, description="Connections to same service in past 2 seconds"
    )
    serror_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    rerror_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    same_srv_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    diff_srv_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    dst_host_count: int = Field(default=1, ge=1)
    dst_host_srv_count: int = Field(default=1, ge=1)
    dst_host_same_srv_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    dst_host_diff_srv_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    protocol_type: str = Field(default="tcp", description="tcp, udp, or icmp")
    service: str = Field(default="http", description="Network service (http, ftp, etc.)")
    flag: str = Field(default="SF", description="Connection status flag")

    @field_validator("protocol_type")
    @classmethod
    def validate_protocol(cls, v: str) -> str:
        allowed = {"tcp", "udp", "icmp"}
        if v.lower() not in allowed:
            raise ValueError(f"protocol_type must be one of {allowed}")
        return v.lower()


class BatchRequest(BaseModel):
    """A batch of network flows to classify in one call."""

    flows: list[NetworkFlow] = Field(..., min_length=1, description="Network flows to classify")

    @field_validator("flows")
    @classmethod
    def validate_batch_size(cls, v: list[NetworkFlow]) -> list[NetworkFlow]:
        maximum = get_settings().max_batch_size
        if len(v) > maximum:
            raise ValueError(f"Batch of {len(v)} exceeds the maximum of {maximum}")
        return v


class PredictionResponse(BaseModel):
    correlation_id: str
    predicted_class: str
    is_attack: bool
    confidence: float
    class_probabilities: dict[str, float]
    threat_context: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class MetricsResponse(BaseModel):
    model_metrics: dict[str, Any]
    drift_reports: list[dict[str, Any]]
    prediction_count: int


# ──────────────────────────── API Endpoints ───────────────────────────────────


@app.get("/api/v1/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Return service liveness and model status."""
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        version="1.0.0",
    )


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    summary="Classify a network flow as normal or an attack type",
)
async def predict_endpoint(
    flow: NetworkFlow,
    request: Request,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """Classify a network connection flow.

    Returns the predicted class (normal / dos / probe / r2l / u2r),
    whether it is an attack, confidence score, and optional CVE context.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    correlation_id = request.state.correlation_id
    flow_dict = flow.model_dump()

    engineer = NetworkFeatureEngineer()
    features = engineer.transform([flow_dict])

    result = predict(_model, features)

    # Update rolling reference window
    for key in ("src_bytes", "dst_bytes", "duration"):
        _reference_data[key].append(flow_dict[key])
        if len(_reference_data[key]) > 2000:
            _reference_data[key] = _reference_data[key][-2000:]
    _reference_data["confidence"].append(result["confidence"])

    log_prediction(db, correlation_id, flow_dict, result)

    threat_ctx: str | None = None
    if result["is_attack"] and _retriever is not None:
        ctx = _retriever.search(result["predicted_class"], top_k=1)
        threat_ctx = ctx[0]["text"] if ctx else None

    return PredictionResponse(
        correlation_id=correlation_id,
        predicted_class=result["predicted_class"],
        is_attack=bool(result["is_attack"]),
        confidence=result["confidence"],
        class_probabilities=result["class_probabilities"],
        threat_context=threat_ctx,
    )


@app.post(
    "/api/v1/predict/batch",
    tags=["inference"],
    summary="Classify many network flows in a single request",
)
async def predict_batch_endpoint(
    batch: BatchRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Classify up to `MAX_BATCH_SIZE` flows in one call.

    Returns one result per input flow, in the order submitted, plus a summary
    counting how many were judged malicious.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    correlation_id = request.state.correlation_id
    flows = [f.model_dump() for f in batch.flows]

    engineer = NetworkFeatureEngineer()
    features = engineer.transform(flows)

    results: list[dict[str, Any]] = []
    for i, flow_dict in enumerate(flows):
        result = predict(_model, features[i : i + 1])
        log_prediction(db, correlation_id, flow_dict, result)
        results.append(result)

    attacks = sum(r["is_attack"] for r in results)
    return {
        "correlation_id": correlation_id,
        "count": len(results),
        "attacks_detected": int(attacks),
        "results": results,
    }


@app.get("/api/v1/metrics", response_model=MetricsResponse, tags=["ops"])
async def metrics_endpoint(db: Session = Depends(get_db)) -> MetricsResponse:
    """Return model training metrics and recent drift reports."""
    from app.database import PredictionLog  # noqa: PLC0415

    count = db.query(PredictionLog).count()

    drift = []
    if len(_reference_data.get("src_bytes", [])) >= 10:
        drift = run_full_drift_check(db, _reference_data)

    return MetricsResponse(
        model_metrics=load_metrics(),
        drift_reports=drift,
        prediction_count=count,
    )


@app.get("/api/v1/drift", tags=["ops"])
async def drift_history(db: Session = Depends(get_db), limit: int = 20) -> dict[str, Any]:
    """Return the most recent drift detection reports."""
    return {"reports": get_drift_summary(db, limit=limit)}


@app.get("/api/v1/threats", tags=["intelligence"])
async def threat_search(q: str, top_k: int = 3) -> dict[str, Any]:
    """Search the threat intelligence index for CVE / MITRE ATT&CK context."""
    if _retriever is None:
        raise HTTPException(status_code=503, detail="RAG retriever not ready")
    results = _retriever.search(q, top_k=min(top_k, 10))
    return {"query": q, "results": results}
