"""FastAPI application entry point for Ops-Vision."""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.v1.routes import router as v1_router
from app.config import get_settings
from app.middleware import CorrelationIdMiddleware, RateLimitMiddleware, RequestSizeLimitMiddleware

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title="Ops-Vision",
    description=(
        "SRE ML platform for real-time incident prediction, alert classification, "
        "and performance anomaly detection using XGBoost/LightGBM/RF ensemble, "
        "FAISS runbook RAG, and KS-test drift monitoring."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestSizeLimitMiddleware)
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    max_requests=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds,
)

app.include_router(v1_router)


@app.on_event("startup")
async def startup_event() -> None:
    """Initialise the database and warm up the ML model on startup."""
    logger.info("Ops-Vision %s starting up", __version__)
    try:
        from app.database import create_tables

        create_tables()
    except Exception:
        logger.exception("Database init failed — continuing without DB")

    try:
        from app.api.v1.routes import _get_model

        _get_model()
        logger.info("Model warm-up complete")
    except Exception:
        logger.exception("Model warm-up failed — will retry on first request")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Log shutdown."""
    logger.info("Ops-Vision shutting down")


@app.get("/health", tags=["health"], summary="Root health check")
def root_health():
    """Lightweight liveness probe for load balancers.

    Returns:
        Dict with status 'ok' and application version.
    """
    return {"status": "ok", "version": __version__}


@app.get("/version", tags=["health"], summary="Application version")
def version():
    """Return the application version string.

    Returns:
        Dict with the semantic version.
    """
    return {"version": __version__}


@app.get("/ready", tags=["health"], summary="Readiness probe")
def ready():
    """Kubernetes-style readiness check — fails if the model is not loaded.

    Returns:
        Dict with status 'ready' or raises 503 if the model is unavailable.
    """
    from fastapi import HTTPException

    from app.api.v1.routes import _model

    if _model is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded")
    return {"status": "ready", "version": __version__}
