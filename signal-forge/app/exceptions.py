"""Custom exception classes for Signal-Forge."""

from __future__ import annotations


class SignalForgeError(Exception):
    """Base exception for all Signal-Forge errors."""


class FeatureExtractionError(SignalForgeError):
    """Raised when the feature engineering pipeline fails."""


class ModelNotFoundError(SignalForgeError):
    """Raised when the serialised model file is missing or unreadable."""


class PredictionError(SignalForgeError):
    """Raised when inference fails after features are prepared."""


class DriftDetectionError(SignalForgeError):
    """Raised when the KS-test drift monitor encounters an error."""


class DatabaseError(SignalForgeError):
    """Raised when a database operation fails unexpectedly."""


class ValidationError(SignalForgeError):
    """Raised when input data fails business-rule validation."""
