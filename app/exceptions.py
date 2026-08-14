"""Custom exception hierarchy for Watt-Guard."""

from __future__ import annotations


class WattGuardError(Exception):
    """Base exception for all Watt-Guard application errors."""


class ModelNotLoadedError(WattGuardError):
    """Raised when a prediction is requested but no model is loaded."""


class FeatureValidationError(WattGuardError):
    """Raised when input features fail schema or range validation."""

    def __init__(self, field: str, reason: str) -> None:
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


__all__ = [
    "ConfigurationError",
    "DatabaseError",
    "DriftDetectionError",
    "FeatureValidationError",
    "ModelNotLoadedError",
    "PredictionError",
    "WattGuardError",
]
