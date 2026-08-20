"""Domain exceptions and FastAPI exception handlers."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class LogisticsFlowError(Exception):
    """Base class for all application errors."""

    status_code: int = 500
    detail: str = "Internal error"

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.detail
        super().__init__(self.detail)


class ModelNotLoadedError(LogisticsFlowError):
    """Raised when inference is attempted before the model is available."""

    status_code = 503
    detail = "Model not loaded"


class FeatureExtractionError(LogisticsFlowError):
    """Raised when the feature pipeline cannot transform a request."""

    status_code = 422
    detail = "Feature extraction failed"


class RateLimitExceededError(LogisticsFlowError):
    """Raised when a client exceeds the configured request rate."""

    status_code = 429
    detail = "Rate limit exceeded"


def register_exception_handlers(app: FastAPI) -> None:
    """Attach a JSON handler for every LogisticsFlowError subclass."""

    @app.exception_handler(LogisticsFlowError)
    async def _handle(request: Request, exc: LogisticsFlowError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "n/a")
        logger.warning("[%s] %s: %s", request_id, type(exc).__name__, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": type(exc).__name__,
                "detail": exc.detail,
                "request_id": request_id,
            },
        )
