"""FastAPI application for Property-Sage valuation API.

Exposes versioned REST endpoints for property price and rental yield
prediction, model monitoring, and drift detection.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.features import NEIGHBORHOODS, PROPERTY_TYPES, property_to_dataframe
from app.health import deep_health_check
from app.retrain_endpoint import router as retrain_router
from app.model import get_metrics, load_models, predict
from app.monitoring import check_prediction_drift, get_prediction_stats, log_prediction
from rag.retriever import neighbourhood_summary, retrieve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_price_model = None
_rental_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise DB and load models on startup; log shutdown."""
    global _price_model, _rental_model
    init_db()
    _price_model, _rental_model = load_models()
    logger.info("Property-Sage API started — models loaded")
    yield
    logger.info("Property-Sage API shutting down")


app = FastAPI(
    title="Property-Sage API",
    description=(
        "Real estate property valuation and rental yield prediction API. "
        "Uses an XGBoost–LightGBM–RandomForest ensemble with automated drift detection."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(retrain_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- simple in-process rate limiter ---
_request_times: dict[str, list[float]] = {}
_RATE_LIMIT = 60
_RATE_WINDOW = 60.0


def _check_rate_limit(client_ip: str) -> bool:
    """Return False if client has exceeded the request rate limit."""
    now = time.time()
    window_start = now - _RATE_WINDOW
    times = [t for t in _request_times.get(client_ip, []) if t > window_start]
    _request_times[client_ip] = times
    if len(times) >= _RATE_LIMIT:
        return False
    times.append(now)
    _request_times[client_ip] = times
    return True


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Attach a correlation ID and response-time header to every response."""
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    client_ip = request.client.host if request.client else "unknown"

    if not _check_rate_limit(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests"},
            headers={"X-Correlation-ID": correlation_id},
        )

    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 2)
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Response-Time-Ms"] = str(duration_ms)
    logger.info(
        "method=%s path=%s status=%d duration_ms=%s correlation_id=%s client=%s",
        request.method, request.url.path, response.status_code,
        duration_ms, correlation_id, client_ip,
    )
    return response


# --- Request / Response schemas ---

class PropertyRequest(BaseModel):
    """Input schema for a property valuation request."""

    bedrooms: int = Field(..., ge=1, le=20, description="Number of bedrooms (1–20).")
    bathrooms: float = Field(..., ge=0.5, le=15.0, description="Number of bathrooms.")
    sqft: float = Field(..., ge=100, le=50_000, description="Total interior square footage.")
    lot_size: float | None = Field(None, ge=0, description="Lot size in square feet (optional).")
    year_built: int = Field(..., ge=1800, le=2024, description="Year the property was constructed.")
    neighborhood: str = Field(..., description=f"One of: {', '.join(NEIGHBORHOODS)}.")
    property_type: str = Field(..., description=f"One of: {', '.join(PROPERTY_TYPES)}.")

    @field_validator("neighborhood")
    @classmethod
    def validate_neighborhood(cls, v: str) -> str:
        """Normalise and validate the neighborhood field."""
        v = v.lower().strip()
        if v not in NEIGHBORHOODS:
            raise ValueError(f"neighborhood must be one of {NEIGHBORHOODS}")
        return v

    @field_validator("property_type")
    @classmethod
    def validate_property_type(cls, v: str) -> str:
        """Normalise and validate the property_type field."""
        v = v.lower().strip()
        if v not in PROPERTY_TYPES:
            raise ValueError(f"property_type must be one of {PROPERTY_TYPES}")
        return v


class PredictionResponse(BaseModel):
    """Output schema for a successful valuation prediction."""

    request_id: str = Field(..., description="Unique identifier for this inference request.")
    predicted_price: float = Field(..., description="Estimated market value in USD.")
    predicted_rental_yield: float = Field(..., description="Estimated gross rental yield (0–1).")
    estimated_annual_rental: float = Field(..., description="Estimated annual rental income in USD.")
    estimated_monthly_rental: float = Field(..., description="Estimated monthly rental income in USD.")
    neighborhood: str
    property_type: str


# --- Endpoints ---

@app.get(
    "/api/v1/health",
    tags=["System"],
    summary="Service liveness check",
    response_description="Service status and version",
)
async def health() -> dict[str, str]:
    """Return service health status.

    Returns:
        Dict with status, service name, and version string.
    """
    return {"status": "ok", "service": "property-sage", "version": "1.0.0"}


@app.get(
    "/api/v1/health/deep",
    tags=["System"],
    summary="Deep health check with DB and model diagnostics",
)
async def health_deep(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Run a deep health check inspecting DB, model files, and in-memory state.

    Args:
        db: Injected database session.

    Returns:
        Dict with per-component health statuses and an overall status field.
    """
    return deep_health_check(
        db=db,
        price_model_loaded=_price_model is not None,
        rental_model_loaded=_rental_model is not None,
    )


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict property price and rental yield",
    response_description="Valuation and rental yield estimates",
)
async def predict_endpoint(
    body: PropertyRequest,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """Predict market price and rental yield for a given property.

    Args:
        body: Property attributes (bedrooms, sqft, neighbourhood, etc.).
        db: Injected database session.

    Returns:
        PredictionResponse with price, yield, and rental estimates.

    Raises:
        HTTPException 503: If models have not been loaded yet.
    """
    if _price_model is None or _rental_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded — service starting up",
        )

    request_id = str(uuid.uuid4())
    X = property_to_dataframe(body.model_dump())
    output = predict(_price_model, _rental_model, X)
    log_prediction(db, request_id, body.model_dump(), output)

    return PredictionResponse(
        request_id=request_id,
        neighborhood=body.neighborhood,
        property_type=body.property_type,
        **output,
    )


@app.get(
    "/api/v1/metrics",
    tags=["Monitoring"],
    summary="Model performance metrics and prediction statistics",
)
async def metrics_endpoint(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return combined model training metrics and live prediction statistics.

    Args:
        db: Injected database session.

    Returns:
        Dict with model_performance (R², RMSE) and prediction_stats (counts, averages).
    """
    model_metrics = get_metrics()
    pred_stats = get_prediction_stats(db)
    return {"model_performance": model_metrics, "prediction_stats": pred_stats}


@app.post(
    "/api/v1/drift-check",
    tags=["Monitoring"],
    summary="Run KS-test drift detection on recent predictions",
)
async def drift_check(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Execute KS-test drift detection on the last 24 hours of predictions.

    Args:
        db: Injected database session.

    Returns:
        Dict with drift_detected flag and per-feature KS statistics.
    """
    return check_prediction_drift(db)


@app.get(
    "/api/v1/neighbourhood/{name}",
    tags=["RAG"],
    summary="Get neighbourhood market intelligence report",
)
async def neighbourhood_report(name: str) -> dict[str, str]:
    """Return the RAG-retrieved market report for a specific neighbourhood.

    Args:
        name: Neighbourhood identifier (e.g. 'downtown', 'suburb').

    Returns:
        Dict with neighbourhood name and market report text.
    """
    report = neighbourhood_summary(name.lower().strip())
    return {"neighbourhood": name, "market_report": report}


@app.get(
    "/api/v1/neighbourhood-search",
    tags=["RAG"],
    summary="Semantic neighbourhood search",
)
async def neighbourhood_search(q: str, top_k: int = 2) -> dict[str, Any]:
    """Retrieve the most relevant neighbourhood documents for a free-text query.

    Args:
        q: Free-text query (e.g. 'high rental yield student area').
        top_k: Number of results to return (default 2).

    Returns:
        Dict with query string and list of matching neighbourhood docs with scores.
    """
    results = retrieve(q, top_k=min(top_k, 5))
    return {"query": q, "results": results}


@app.get("/", tags=["System"], summary="API root")
async def root() -> dict[str, str]:
    """Return a welcome message with a link to the interactive docs."""
    return {"message": "Property-Sage API — see /docs for usage"}
