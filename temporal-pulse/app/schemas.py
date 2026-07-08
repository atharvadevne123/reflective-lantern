"""Pydantic request and response schemas for Temporal-Pulse API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SensorReading(BaseModel):
    """A single multivariate sensor reading."""

    sensor_id: str = Field(..., min_length=1, max_length=64, description="Unique sensor identifier")
    timestamp: datetime = Field(..., description="ISO-8601 timestamp of the reading")
    values: dict[str, float] = Field(
        ...,
        description="Key-value pairs of channel name to numeric value",
        min_length=1,
    )

    @field_validator("values")
    @classmethod
    def values_must_be_finite(cls, v: dict[str, float]) -> dict[str, float]:
        """Reject NaN and infinite values."""
        import math

        for key, val in v.items():
            if not math.isfinite(val):
                raise ValueError(f"Channel '{key}' contains non-finite value: {val}")
        return v


class BatchSensorReadings(BaseModel):
    """Batch of sensor readings for bulk scoring."""

    readings: list[SensorReading] = Field(..., min_length=1, max_length=10_000)
    horizon: int = Field(default=5, ge=1, le=100, description="Forecast horizon in time steps")


class AnomalyResult(BaseModel):
    """Anomaly detection result for a single reading."""

    sensor_id: str
    timestamp: datetime
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    is_anomaly: bool
    threshold: float


class ForecastResult(BaseModel):
    """Forecast result for a sensor channel."""

    sensor_id: str
    channel: str
    horizon: int
    predicted_values: list[float]
    confidence_lower: list[float]
    confidence_upper: list[float]
    model_version: str


class BatchAnomalyResponse(BaseModel):
    """Response for batch anomaly detection."""

    results: list[AnomalyResult]
    total_readings: int
    anomaly_count: int
    processing_time_ms: float


class DriftReport(BaseModel):
    """KS-test drift detection report."""

    feature: str
    ks_statistic: float | None = None
    p_value: float | None = None
    drift_detected: bool
    reference_mean: float | None = None
    current_mean: float | None = None
    reference_std: float | None = None
    current_std: float | None = None
    reason: str | None = None


class MetricsResponse(BaseModel):
    """System-wide prediction metrics."""

    total_predictions: int
    anomaly_score_mean: float | None = None
    anomaly_score_p95: float | None = None
    anomaly_score_p99: float | None = None
    latency_mean_ms: float | None = None
    latency_p95_ms: float | None = None
    drift_features_tested: int | None = None


class HealthResponse(BaseModel):
    """API health check response."""

    status: str
    version: str
    model_loaded: bool
    db_connected: bool


class VersionResponse(BaseModel):
    """Version information."""

    api_version: str
    model_version: str
    build: str = "temporal-pulse-1.0.0"


class TrainRequest(BaseModel):
    """Request to trigger model training."""

    readings: list[SensorReading] = Field(..., min_length=50)
    contamination: float = Field(default=0.05, ge=0.01, le=0.5)
    horizon: int = Field(default=5, ge=1, le=100)


class TrainResponse(BaseModel):
    """Training result summary."""

    model_version: str
    samples_trained: int
    features_used: int
    metrics: dict[str, Any]
