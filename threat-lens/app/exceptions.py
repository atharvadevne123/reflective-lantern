"""Domain exceptions for Threat-Lens."""


class ThreatLensError(Exception):
    """Base class for every error raised by this service."""


class ModelNotLoadedError(ThreatLensError):
    """Raised when inference is attempted before a model is available."""

    def __init__(self, message: str = "Model is not loaded") -> None:
        super().__init__(message)


class FeatureMismatchError(ThreatLensError):
    """Raised when input feature count does not match what the model expects."""

    def __init__(self, expected: int, received: int) -> None:
        self.expected = expected
        self.received = received
        super().__init__(
            f"Model expects {expected} features but received {received}. "
            "Retrain the model after changing FEATURE_NAMES."
        )


class RetrieverNotReadyError(ThreatLensError):
    """Raised when the threat intelligence index has not been built."""

    def __init__(self, message: str = "Threat intelligence index is not built") -> None:
        super().__init__(message)


class BatchTooLargeError(ThreatLensError):
    """Raised when a batch request exceeds the configured maximum."""

    def __init__(self, size: int, maximum: int) -> None:
        self.size = size
        self.maximum = maximum
        super().__init__(f"Batch of {size} exceeds the maximum of {maximum}")
