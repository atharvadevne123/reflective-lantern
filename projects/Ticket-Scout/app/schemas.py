"""Shared Pydantic schemas for request/response validation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Service health response."""

    status: str = Field(..., examples=["healthy"])
    uptime_seconds: float
    model_version: str
    models_loaded: bool


class MetricsResponse(BaseModel):
    """Aggregated model and service metrics."""

    model_metrics: dict
    prediction_volume_24h: int
    predicted_breach_count_24h: int
    drift_available: bool


class DriftResult(BaseModel):
    """Single feature drift result."""

    feature: str
    ks_statistic: float
    p_value: float
    drift_detected: bool


class DriftResponse(BaseModel):
    """Drift check response for all monitored features."""

    features_checked: int
    drift_detected_count: int
    results: list[DriftResult]


class RetrainResponse(BaseModel):
    """Model retrain response."""

    status: str
    metrics: dict
