"""Pydantic v2 request/response schemas for Ticket-Oracle API.

Centralised schema definitions so main.py stays lean and schemas
can be imported by tests without pulling in the full application.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator


class TicketPayload(BaseModel):
    ticket_id: Annotated[str, Field(min_length=1, max_length=64)]
    description: Annotated[str, Field(min_length=10, max_length=4000)]
    department: str
    incident_type: str
    channel: str
    customer_tier: str
    product_area: str
    open_tickets_count: Annotated[int, Field(ge=0, le=10_000)]
    agent_load: Annotated[float, Field(ge=0.0, le=1.0)]
    hour_of_day: Annotated[int, Field(ge=0, le=23)]
    day_of_week: Annotated[int, Field(ge=0, le=6)]
    ticket_age_minutes: Annotated[float, Field(ge=0.0)]
    same_user_last_7d: Annotated[int, Field(ge=0)]

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v: str) -> str:
        allowed = {"email", "phone", "chat", "portal"}
        if v not in allowed:
            raise ValueError(f"channel must be one of {sorted(allowed)}, got {v!r}")
        return v

    @field_validator("customer_tier")
    @classmethod
    def validate_customer_tier(cls, v: str) -> str:
        allowed = {"Bronze", "Standard", "Silver", "Gold"}
        if v not in allowed:
            raise ValueError(f"customer_tier must be one of {sorted(allowed)}, got {v!r}")
        return v


class BatchTicketPayload(BaseModel):
    tickets: Annotated[list[TicketPayload], Field(min_length=1, max_length=100)]


class PredictionResponse(BaseModel):
    ticket_id: str
    priority: str
    priority_confidence: float
    sla_breach_risk: float
    resolution_hours_estimate: float
    similar_tickets: list[dict[str, Any]] = Field(default_factory=list)
    cached: bool = False


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class TrainResponse(BaseModel):
    status: str
    auc_roc: float
    n_samples: int


class MetricsResponse(BaseModel):
    request_count: int
    error_count: int
    cache_hits: int
    cache_misses: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
