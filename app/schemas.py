"""Pydantic request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class EnergyReadingIn(BaseModel):
    """Input schema for a single energy reading."""

    building_id: str = Field(..., min_length=1, max_length=64, description="Unique building identifier")
    timestamp: datetime = Field(..., description="Reading timestamp (ISO-8601)")
    hour: int = Field(..., ge=0, le=23, description="Hour of day 0-23")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week 0=Mon 6=Sun")
    month: int = Field(..., ge=1, le=12, description="Month 1-12")
    temperature_c: float = Field(..., ge=-40.0, le=60.0, description="Outside temperature in Celsius")
    humidity_pct: float = Field(..., ge=0.0, le=100.0, description="Relative humidity 0-100")
    occupancy: int = Field(..., ge=0, le=10000, description="Number of occupants")
    hvac_state: int = Field(..., ge=0, le=1, description="HVAC on=1 off=0")
    consumption_kwh: float = Field(0.0, ge=0.0, description="Current consumption (used for lag features)")

    @field_validator("building_id")
    @classmethod
    def building_id_alphanumeric(cls, v: str) -> str:
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("building_id must be alphanumeric with hyphens/underscores only")
        return v


class PredictResponse(BaseModel):
    """Forecasting model output for a single building reading."""

    building_id: str
    timestamp: datetime
    predicted_kwh: float
    model_version: str
    latency_ms: float


class AnomalyRequest(BaseModel):
    """Input for anomaly detection."""

    building_id: str = Field(..., min_length=1, max_length=64)
    timestamp: datetime
    consumption_kwh: float = Field(..., ge=0.0)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    temperature_c: float = Field(20.0, ge=-40.0, le=60.0)
    humidity_pct: float = Field(50.0, ge=0.0, le=100.0)
    occupancy: int = Field(0, ge=0, le=10000)
    hvac_state: int = Field(0, ge=0, le=1)


class AnomalyResponse(BaseModel):
    """Anomaly detection result for a single consumption reading."""

    building_id: str
    timestamp: datetime
    consumption_kwh: float
    anomaly_score: float
    is_anomaly: bool
    severity: str
    latency_ms: float


class DriftRequest(BaseModel):
    """Request body for the /drift endpoint."""

    current_values: list[float] = Field(..., min_length=10, description="Current window of consumption readings")
    reference_values: list[float] | None = Field(None, description="Reference distribution (uses global if omitted)")


class DriftResponse(BaseModel):
    """KS-test drift detection result."""

    ks_statistic: float
    p_value: float
    drift_detected: bool
    message: str


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    model_loaded: bool
    anomaly_model_loaded: bool
    version: str


class MetricsResponse(BaseModel):
    """Aggregated monitoring metrics response."""
    total_predictions: int
    total_anomalies_flagged: int
    total_drift_events: int
    reference_window_size: int
    model_metrics: dict[str, Any]


class BatchPredictRequest(BaseModel):
    readings: list[EnergyReadingIn] = Field(..., min_length=1, max_length=100)
