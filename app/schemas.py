"""Pydantic request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

_MAX_SQFT = 50_000
_MAX_BEDROOMS = 20
_MAX_BATHROOMS = 20.0
_MIN_YEAR = 1800
_MAX_YEAR = 2026
_BATCH_MAX = 100
_MAX_TOP_K = 20


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

    sqft: float = Field(..., gt=0, le=_MAX_SQFT, description="Total living area in square feet")
    bedrooms: int = Field(..., ge=1, le=_MAX_BEDROOMS)
    bathrooms: float = Field(..., ge=0.5, le=_MAX_BATHROOMS)
    lot_size: float = Field(0.0, ge=0, description="Lot size in square feet")
    year_built: int = Field(..., ge=_MIN_YEAR, le=_MAX_YEAR)
    renovation_year: int | None = Field(None, ge=_MIN_YEAR, le=_MAX_YEAR)
    condition_score: float = Field(5.0, ge=1.0, le=10.0)
    zipcode: str = Field(..., min_length=5, max_length=10)
    city: str = Field("", max_length=100)
    state: str = Field("", max_length=50)
    school_score: float = Field(5.0, ge=0.0, le=10.0)
    transit_score: float = Field(5.0, ge=0.0, le=10.0)
    walkability_score: float = Field(5.0, ge=0.0, le=10.0)
    crime_rate: float = Field(0.3, ge=0.0, le=1.0)
    median_neighborhood_price: float = Field(300_000.0, gt=0)
    median_price_per_sqft: float = Field(200.0, gt=0)
    avg_rental_yield: float = Field(0.06, ge=0.0, le=1.0)
    listing_days: int = Field(30, ge=0, le=3650)
    list_price: float | None = Field(None, gt=0)

    @field_validator("renovation_year")
    @classmethod
    def building_id_alphanumeric(cls, v: str) -> str:
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("building_id must be alphanumeric with hyphens/underscores only")
        return v


class PredictResponse(BaseModel):
    building_id: str
    timestamp: datetime
    predicted_kwh: float
    model_version: str
    latency_ms: float


class AnomalyRequest(BaseModel):
    """Input for anomaly detection."""

    properties: list[PropertyInput] = Field(..., min_length=1, max_length=_BATCH_MAX)


class AnomalyResponse(BaseModel):
    building_id: str
    timestamp: datetime
    consumption_kwh: float
    anomaly_score: float
    is_anomaly: bool
    severity: str
    latency_ms: float


class ComparableRequest(BaseModel):
    """Request schema for comparable-property search via FAISS."""

    property: PropertyInput
    top_k: int = Field(5, ge=1, le=_MAX_TOP_K)


class DriftResponse(BaseModel):
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
