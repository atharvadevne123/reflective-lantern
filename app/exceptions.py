"""Custom exception hierarchy for Watt-Guard."""

from __future__ import annotations


class WattGuardError(Exception):
    """Base exception for all Watt-Guard application errors."""


class ModelNotLoadedError(WattGuardError):
    """Raised when a prediction is requested but no model is loaded."""


class FeatureValidationError(WattGuardError):
    """Raised when input features fail schema or range validation."""

    def __init__(self, field: str, reason: str) -> None:
        """Attach field/reason metadata and format a human-readable message.

        Args:
            field: Name of the feature field that failed validation.
            reason: Human-readable explanation of the validation failure.
        """
        self.field = field
        self.reason = reason
        super().__init__(f"Feature validation failed for '{field}': {reason}")


class DriftDetectionError(WattGuardError):
    """Raised when drift detection cannot complete due to insufficient data."""


class DatabaseError(WattGuardError):
    """Raised for unrecoverable database operation failures."""


class ConfigurationError(WattGuardError):
    """Raised when required configuration is missing or invalid."""


class PredictionError(WattGuardError):
    """Raised when the model pipeline fails to produce a prediction."""


class RateLimitExceededError(WattGuardError):
    """Raised when a client exceeds the configured request rate limit."""

    def __init__(self, limit: int, retry_after_seconds: int = 60) -> None:
        """Store limit/retry metadata and format the exception message.

        Args:
            limit: The rate limit that was exceeded (requests per minute).
            retry_after_seconds: Suggested wait time before retrying (default 60).
        """
        self.limit = limit
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"Rate limit of {limit} req/min exceeded; retry after {retry_after_seconds}s",
        )


class ExternalServiceError(WattGuardError):
    """Raised when a downstream/external service is unreachable or errors out."""

    def __init__(self, service: str, reason: str) -> None:
        """Store service/reason metadata and format the exception message.

        Args:
            service: Identifier or name of the external service that failed.
            reason: Description of the failure (status code, error message, etc.).
        """
        self.service = service
        self.reason = reason
        super().__init__(f"External service {service!r} failed: {reason}")


__all__ = [
    "ConfigurationError",
    "DatabaseError",
    "DriftDetectionError",
    "ExternalServiceError",
    "FeatureValidationError",
    "ModelNotLoadedError",
    "PredictionError",
    "RateLimitExceededError",
    "WattGuardError",
]
