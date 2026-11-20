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
        """Initialise with a fixed message and MODEL_NOT_LOADED code."""
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


class RateLimitExceededError(SocialPulseError):
    """Raised when a client exceeds the configured request rate."""

    def __init__(self, client_ip: str = "unknown") -> None:
        """Create a rate limit exceeded error.

        Args:
            client_ip: The IP address of the client that was rate-limited.
        """
        super().__init__(
            f"Rate limit exceeded for client {client_ip}",
            code="RATE_LIMITED",
        )
        self.client_ip = client_ip


class InvalidFeatureVectorError(SocialPulseError):
    """Raised when a feature vector has incorrect dimensionality."""

    def __init__(self, expected: int, got: int) -> None:
        """Create an invalid feature vector error.

        Args:
            expected: Expected vector dimensionality.
            got: Actual dimensionality received.
        """
        super().__init__(
            f"Invalid feature vector: expected dim={expected}, got dim={got}",
            code="INVALID_FEATURE_VECTOR",
        )
        self.expected = expected
        self.got = got
