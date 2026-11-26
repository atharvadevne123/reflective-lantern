"""Pydantic request/response schemas with full validation for Support-Sage."""

from __future__ import annotations

import uuid
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class TicketRequest(BaseModel):
    """Incoming ticket for triage."""

    ticket_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject: Annotated[str, Field(min_length=3, max_length=512, description="Ticket subject line")]
    description: Annotated[str, Field(min_length=10, max_length=4096, description="Full description")]

    @field_validator("subject", "description")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class PriorityResponse(BaseModel):
    """Triage result returned to the caller."""

    ticket_id: str
    priority: str
    confidence: float
    priority_probabilities: dict[str, float]
    resolution_hours_estimate: float
    model_version: str = "1.0.0"


class HealthResponse(BaseModel):
    """Service liveness check response."""

    status: str
    model_loaded: bool
    version: str


class RetrainResponse(BaseModel):
    """Response for POST /api/v1/train."""

    status: str
    metrics: dict[str, object]
