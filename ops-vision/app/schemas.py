"""Pydantic request/response schemas for Ops-Vision API."""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class MetricsPayload(BaseModel):
    """Input metrics for a single prediction request."""

    service_name: str = Field(..., min_length=1, max_length=128)
    cpu_usage_pct: float = Field(..., ge=0.0, le=100.0)
    memory_usage_pct: float = Field(..., ge=0.0, le=100.0)
    error_rate_per_min: float = Field(..., ge=0.0)
    latency_p99_ms: float = Field(..., ge=0.0)
    request_rate_per_sec: float = Field(..., ge=0.0)
    disk_io_util_pct: float = Field(..., ge=0.0, le=100.0)

    @field_validator("service_name")
    @classmethod
    def strip_service_name(cls, v: str) -> str:
        """Strip whitespace from service name."""
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "example": {
                "service_name": "payments-api",
                "cpu_usage_pct": 85.0,
                "memory_usage_pct": 88.0,
                "error_rate_per_min": 62.0,
                "latency_p99_ms": 1450.0,
                "request_rate_per_sec": 45.0,
                "disk_io_util_pct": 80.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response payload for a prediction."""

    service_name: str
    predicted_incident: bool
    predicted_severity: str | None
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    runbook_hint: str | None = None
    timestamp: datetime


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    reference_window_size: int
    current_window_size: int
    version: str


class MetricsResponse(BaseModel):
    """Aggregated operational metrics for monitoring dashboards."""

    total_predictions: int
    incident_count: int
    incident_rate: float
    drift_alerts_24h: int
    avg_confidence: float


class RunbookSearchRequest(BaseModel):
    """Request to search runbooks via FAISS similarity."""

    query: str = Field(..., min_length=3, max_length=512)
    top_k: int = Field(default=3, ge=1, le=10)


class RunbookResult(BaseModel):
    """A single runbook returned by FAISS search."""

    title: str
    content: str
    score: float
    category: str


class BatchPredictRequest(BaseModel):
    """A batch of telemetry observations to score in one call."""

    items: list["MetricsPayload"] = Field(..., min_length=1, max_length=500)


class IncidentRecord(BaseModel):
    """A persisted incident row returned by the incidents listing."""

    id: int
    service_name: str
    cpu_usage_pct: float
    memory_usage_pct: float
    error_rate_per_min: float
    latency_p99_ms: float
    request_rate_per_sec: float
    disk_io_util_pct: float
    is_incident: bool
    severity: str | None
    created_at: datetime | None

    model_config = {"from_attributes": True}


class ForecastPoint(BaseModel):
    """A single forecasted incident-rate value at a future timestamp."""

    timestamp: datetime
    value: float = Field(..., ge=0.0)
    lower_bound: float = Field(..., ge=0.0)
    upper_bound: float = Field(..., ge=0.0)


class DriftStatusResponse(BaseModel):
    """Response summarising the latest drift detection results."""

    checked_at: datetime | None
    features_drifted: list[str]
    features_stable: list[str]
    total_features: int
