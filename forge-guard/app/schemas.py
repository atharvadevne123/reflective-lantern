"""Pydantic schemas for Forge-Guard request/response validation.

All schemas use strict Pydantic v2 validation. Field constraints mirror the
physical plausibility ranges enforced by the SensorInput model in main.py.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BatchSensorInput(BaseModel):
    """Validated payload for a batch inference request (up to 100 rows).

    Attributes:
        readings: List of sensor reading dicts. Each dict must contain the
            same keys as SensorInput. Between 1 and 100 readings inclusive.
    """

    readings: list[dict[str, float]] = Field(
        ..., min_length=1, max_length=100, description="List of sensor readings"
    )


class BatchPredictionResponse(BaseModel):
    """Response envelope for batch predictions.

    Attributes:
        predictions: List of per-reading results, each containing
            ``prediction`` (0/1) and ``defect_probability`` (float).
        count: Number of predictions returned — equals len(predictions).
        model_version: Semantic version of the model that produced results.
    """

    predictions: list[dict[str, Any]]
    count: int
    model_version: str


class RetrainingTriggerResponse(BaseModel):
    """Response from the /api/v1/retrain endpoint.

    Attributes:
        status: Either ``"accepted"`` (pipeline started) or ``"error"``.
        message: Human-readable description of the outcome.
    """

    status: str
    message: str


class DriftCheckResult(BaseModel):
    """Single-feature KS-test drift result.

    Attributes:
        feature: Sensor column name that was tested.
        ks_statistic: KS test statistic in [0, 1].
        p_value: p-value of the KS test; p < 0.05 indicates drift.
        drift_detected: True when p_value < configured threshold (default 0.05).
    """

    feature: str
    ks_statistic: float
    p_value: float
    drift_detected: bool


class ModelSummaryResponse(BaseModel):
    """Aggregate statistics for a specific model version.

    Attributes:
        model_version: The model version these stats describe.
        total: Total number of predictions recorded.
        defects: Count of predictions classified as defects.
        defect_rate: Fraction of total predictions classified as defects.
        avg_defect_probability: Mean raw defect probability across all predictions.
    """

    model_version: str
    total: int
    defects: int = 0
    defect_rate: float = 0.0
    avg_defect_probability: float = 0.0
