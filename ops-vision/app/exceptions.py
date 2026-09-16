"""Custom exception types for Ops-Vision."""


class OpsVisionError(Exception):
    """Base class for all Ops-Vision application errors."""


class ModelNotLoadedError(OpsVisionError):
    """Raised when a prediction is attempted before the model is loaded."""

    def __init__(self) -> None:
        """Raise with a message directing callers to load artifacts first."""
        super().__init__("ML model is not loaded. Call _load_artifacts() first.")


class FeatureEngineeringError(OpsVisionError):
    """Raised when the feature pipeline transform fails."""

    def __init__(self, detail: str = "Feature engineering failed") -> None:
        """Store detail and format the exception message.

        Args:
            detail: Description of the feature engineering failure.
        """
        self.detail = detail
        super().__init__(detail)


class DriftMonitorError(OpsVisionError):
    """Raised when drift detection cannot run."""

    def __init__(self, reason: str) -> None:
        """Store the failure reason and format the exception message.

        Args:
            reason: Description of why drift monitoring could not run.
        """
        self.reason = reason
        super().__init__(f"Drift monitor error: {reason}")


class RunbookIndexError(OpsVisionError):
    """Raised when the runbook FAISS index operation fails."""

    def __init__(self, reason: str) -> None:
        """Store the failure reason and format the exception message.

        Args:
            reason: Description of the FAISS index operation failure.
        """
        self.reason = reason
        super().__init__(f"Runbook index error: {reason}")


__all__ = [
    "DriftMonitorError",
    "FeatureEngineeringError",
    "ModelNotLoadedError",
    "OpsVisionError",
    "RunbookIndexError",
]
