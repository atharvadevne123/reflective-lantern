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
