"""Custom exception types for Ops-Vision."""


class OpsVisionError(Exception):
    """Base class for all Ops-Vision application errors."""


class ModelNotLoadedError(OpsVisionError):
    """Raised when a prediction is attempted before the model is loaded."""

    def __init__(self) -> None:
        super().__init__("ML model is not loaded. Call _load_artifacts() first.")


class FeatureEngineeringError(OpsVisionError):
    """Raised when the feature pipeline transform fails."""

    def __init__(self, detail: str = "Feature engineering failed") -> None:
        self.detail = detail
        super().__init__(detail)


class DriftMonitorError(OpsVisionError):
    """Raised when drift detection cannot run."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Drift monitor error: {reason}")


class RunbookIndexError(OpsVisionError):
    """Raised when the runbook FAISS index operation fails."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Runbook index error: {reason}")
