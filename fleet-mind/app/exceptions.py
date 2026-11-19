"""Custom exception types for Fleet-Mind."""

from __future__ import annotations


class FleetMindError(Exception):
    """Base exception for all Fleet-Mind errors."""


class ModelNotReadyError(FleetMindError):
    """Raised when a prediction is requested before the model is loaded."""

    def __init__(self, message: str = "Inference model is not ready") -> None:
        super().__init__(message)


class PredictionError(FleetMindError):
    """Raised when the ML model encounters an error during inference.

    Args:
        vehicle_id: The identifier of the vehicle that triggered the error.
        cause: The underlying exception.
    """

    def __init__(self, vehicle_id: str, cause: Exception) -> None:
        self.vehicle_id = vehicle_id
        self.cause = cause
        super().__init__(f"Prediction failed for vehicle {vehicle_id!r}: {cause}")


class DriftCheckError(FleetMindError):
    """Raised when the KS-test drift check cannot be completed."""


class ModelTrainingError(FleetMindError):
    """Raised when model training fails."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        self.cause = cause
        super().__init__(message)


class InvalidReadingError(FleetMindError):
    """Raised when a telemetry reading fails validation.

    Args:
        vehicle_id: Vehicle that submitted the invalid reading.
        field: The field that failed validation.
        value: The offending value.
    """

    def __init__(self, vehicle_id: str, field: str, value: object) -> None:
        self.vehicle_id = vehicle_id
        self.field = field
        self.value = value
        super().__init__(
            f"Invalid reading for vehicle {vehicle_id!r}: {field}={value!r}"
        )


class ThresholdExceededError(FleetMindError):
    """Raised when a metric crosses a critical threshold.

    Args:
        metric: Metric name that exceeded its threshold.
        value: Observed value.
        threshold: The configured threshold.
    """

    def __init__(self, metric: str, value: float, threshold: float) -> None:
        self.metric = metric
        self.value = value
        self.threshold = threshold
        super().__init__(
            f"{metric} value {value:.4f} exceeded threshold {threshold:.4f}"
        )


class RegistryError(FleetMindError):
    """Raised when model registry read/write operations fail."""
