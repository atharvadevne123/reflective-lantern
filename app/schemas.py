"""Pydantic request/response schemas for the Volt-Cast API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    is_weekend: bool = Field(default=False, description="Whether the day is weekend")
    temperature_c: float = Field(..., ge=-30.0, le=55.0, description="Temperature in Celsius")
    humidity_pct: float = Field(..., ge=0.0, le=100.0, description="Relative humidity (0-100)")
    historical_loads: list[float] | None = Field(
        default=None,
        description="Recent historical load values in MW (most recent last)",
    )
    region: str = Field(default="default", max_length=64, description="Grid region identifier")

    @field_validator("historical_loads")
    @classmethod
    def validate_loads(cls, v: list[float] | None) -> list[float] | None:
        if v is not None:
            if len(v) > 1000:
                raise ValueError("historical_loads must not exceed 1000 entries")
            if any(x < 0 or x > 50000 for x in v):
                raise ValueError("load values must be between 0 and 50000 MW")
        return v


class PredictResponse(BaseModel):
    predicted_load_mw: float
    model_version: str
    region: str
    request_id: str
    latency_ms: float


class BatchPredictRequest(BaseModel):
    requests: list[PredictRequest] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    total: int
    batch_latency_ms: float


class ForecastRequest(BaseModel):
    start_hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    is_weekend: bool = False
    temperature_c: float = Field(default=20.0, ge=-30.0, le=55.0)
    humidity_pct: float = Field(default=60.0, ge=0.0, le=100.0)
    horizon_hours: int = Field(default=24, ge=1, le=168)
    region: str = Field(default="default", max_length=64)


class ForecastResponse(BaseModel):
    region: str
    horizon_hours: int
    forecasts: list[dict]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    prediction_count: int
    uptime_seconds: float


class MetricsResponse(BaseModel):
    r2_mean: float | None
    r2_std: float | None
    rmse_mean: float | None
    rmse_std: float | None
    n_samples: int | None
    n_features: int | None
    model_version: str
    prediction_count: int
    recent_latency_p50_ms: float | None
    recent_latency_p95_ms: float | None


class DriftResponse(BaseModel):
    drifts: list[dict]
    summary: dict
    model_version: str
