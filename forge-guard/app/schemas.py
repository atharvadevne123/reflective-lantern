"""Pydantic schemas for Forge-Guard request/response validation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BatchSensorInput(BaseModel):
    """Validated payload for a batch inference request (up to 100 rows)."""

    readings: list[dict[str, float]] = Field(
        ..., min_length=1, max_length=100, description="List of sensor readings"
    )


class BatchPredictionResponse(BaseModel):
    """Response envelope for batch predictions."""

    predictions: list[dict[str, Any]]
    count: int
    model_version: str


class RetrainingTriggerResponse(BaseModel):
    status: str
    message: str
