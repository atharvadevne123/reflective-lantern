"""Custom exception classes for Social-Pulse."""

from __future__ import annotations


class SocialPulseError(Exception):
    """Base exception for all Social-Pulse errors."""

    def __init__(self, message: str, code: str = "INTERNAL_ERROR") -> None:
        """Initialise with a human-readable message and an error code.

        Args:
            message: Descriptive error message.
            code: Short uppercase error code for API responses.
        """
        super().__init__(message)
        self.message = message
        self.code = code


class ModelNotLoadedError(SocialPulseError):
    """Raised when inference is attempted before the model is loaded."""

    def __init__(self) -> None:
        super().__init__(
            "Model is not loaded. Call /train first.",
            code="MODEL_NOT_LOADED",
        )


class FeatureExtractionError(SocialPulseError):
    """Raised when the feature pipeline fails to process a sample."""

    def __init__(self, detail: str) -> None:
        """Create a feature extraction error.

        Args:
            detail: Description of the extraction failure.
        """
        super().__init__(
            f"Feature extraction failed: {detail}",
            code="FEATURE_EXTRACTION_ERROR",
        )


class DatabaseError(SocialPulseError):
    """Raised when a database operation fails."""

    def __init__(self, detail: str) -> None:
        """Create a database error.

        Args:
            detail: Description of the database failure.
        """
        super().__init__(
            f"Database error: {detail}",
            code="DATABASE_ERROR",
        )


class DriftCheckError(SocialPulseError):
    """Raised when the drift detection check encounters an unexpected error."""

    def __init__(self, detail: str) -> None:
        """Create a drift check error.

        Args:
            detail: Description of the drift check failure.
        """
        super().__init__(
            f"Drift check error: {detail}",
            code="DRIFT_CHECK_ERROR",
        )
