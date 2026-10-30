"""Pydantic request/response schemas for Fleet-Mind API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class TelemetryInput(BaseModel):
    """Single vehicle telemetry reading."""

    vehicle_id: str = Field(..., min_length=1, max_length=64, description="Unique vehicle identifier")
    speed_kmh: float = Field(..., ge=0, le=300, description="Current speed in km/h")
    engine_temp_c: float = Field(..., ge=-40, le=200, description="Engine temperature in Celsius")
    oil_pressure_bar: float = Field(..., ge=0, le=10, description="Oil pressure in bar")
    fuel_consumption_lph: float = Field(..., ge=0, le=100, description="Fuel consumption in litres/hour")
    battery_voltage_v: float = Field(..., ge=0, le=20, description="Battery voltage in volts")
    brake_pad_mm: float = Field(..., ge=0, le=20, description="Brake pad thickness in millimetres")
    tire_pressure_bar: float = Field(..., ge=0, le=5, description="Tyre pressure in bar")
    rpm: float = Field(..., ge=0, le=10000, description="Engine RPM")
    coolant_temp_c: float = Field(..., ge=-40, le=150, description="Coolant temperature in Celsius")
    transmission_temp_c: float = Field(..., ge=-40, le=200, description="Transmission temperature in Celsius")

    @field_validator("vehicle_id")
    @classmethod
    def validate_vehicle_id(cls, v: str) -> str:
        """Reject vehicle IDs with path traversal characters."""
        if any(c in v for c in ("/", "\\", "..", "\x00")):
            raise ValueError("vehicle_id contains invalid characters")
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "example": {
                "vehicle_id": "VH-001",
                "speed_kmh": 85.0,
                "engine_temp_c": 92.0,
                "oil_pressure_bar": 3.5,
                "fuel_consumption_lph": 7.2,
                "battery_voltage_v": 12.8,
                "brake_pad_mm": 6.5,
                "tire_pressure_bar": 2.3,
                "rpm": 2800.0,
                "coolant_temp_c": 88.0,
                "transmission_temp_c": 75.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Prediction result returned to the caller."""

    vehicle_id: str
    maintenance_needed: bool
    anomaly_score: float = Field(..., ge=0, le=1)
    anomaly_type: str | None
    remaining_useful_life_km: float | None
    confidence: float = Field(..., ge=0, le=1)
    model_version: str


class BatchTelemetryInput(BaseModel):
    """Batch of telemetry readings."""

    readings: list[TelemetryInput] = Field(..., min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    """Batch prediction result."""

    results: list[PredictionResponse]
    total: int
    maintenance_flagged: int


class HealthResponse(BaseModel):
    """API health check response."""

    status: str
    model_version: str
    db_ok: bool


class MetricsResponse(BaseModel):
    """Model performance metrics."""

    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    roc_auc: float | None = None
    cv_auc_mean: float | None = None
    cv_auc_std: float | None = None
    train_samples: int | None = None
    maintenance_rate: float | None = None


class DriftCheckResponse(BaseModel):
    """Drift analysis result."""

    checked_features: int
    drifted_features: int
    reports: list[dict]
