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

    def __init__(self, detail: str | None = None, *, limit: int = 0, retry_after_seconds: int = 0) -> None:
        self.limit = limit
        self.retry_after_seconds = retry_after_seconds
        if limit and detail is None:
            detail = f"Rate limit exceeded: limit={limit} retry_after_seconds={retry_after_seconds}"
        super().__init__(detail)


# Aliases used by WattGuard / energy-domain tests
WattGuardError = LogisticsFlowError


class PredictionError(LogisticsFlowError):
    """Raised when model inference fails unexpectedly."""

    status_code = 500
    detail = "Prediction failed"


class FeatureValidationError(LogisticsFlowError):
    """Raised when feature input fails schema or range validation."""

    status_code = 422
    detail = "Feature validation failed"

    def __init__(self, field: str = "", reason: str = "") -> None:
        self.field = field
        self.reason = reason
        detail = f"Feature validation failed: field={field!r} reason={reason!r}" if field else self.detail
        super(LogisticsFlowError, self).__init__(detail)
        self.detail = detail


class DatabaseError(LogisticsFlowError):
    """Raised on unrecoverable database operation failures."""

    status_code = 503
    detail = "Database error"


class DriftDetectionError(LogisticsFlowError):
    """Raised when the drift detection routine encounters an error."""

    status_code = 500
    detail = "Drift detection error"


class ConfigurationError(LogisticsFlowError):
    """Raised when a required configuration value is missing or invalid."""

    status_code = 500
    detail = "Configuration error"


class ExternalServiceError(LogisticsFlowError):
    """Raised when a call to an external service fails."""

    status_code = 502
    detail = "External service error"

    def __init__(self, service: str = "", reason: str = "") -> None:
        self.service = service
        self.reason = reason
        detail = f"External service error: service={service!r} reason={reason!r}" if service else self.detail
        super(LogisticsFlowError, self).__init__(detail)
        self.detail = detail


__all__ = [
    "ConfigurationError",
    "DatabaseError",
    "DriftDetectionError",
    "ExternalServiceError",
    "FeatureExtractionError",
    "FeatureValidationError",
    "LogisticsFlowError",
    "ModelNotLoadedError",
    "PredictionError",
    "RateLimitExceededError",
    "WattGuardError",
    "register_exception_handlers",
]


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
