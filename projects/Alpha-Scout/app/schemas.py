"""Pydantic request/response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class OHLCVInput(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")
    close: float = Field(..., gt=0, description="Closing price")
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="Day high price")
    low: float = Field(..., gt=0, description="Day low price")
    volume: float = Field(..., gt=0, description="Trading volume")
    pe_ratio: float | None = Field(None, gt=0, description="Price-to-earnings ratio")
    market_cap: float | None = Field(None, gt=0, description="Market capitalisation in USD")

    @model_validator(mode="after")
    def high_gte_low(self) -> OHLCVInput:
        if self.high < self.low:
            raise ValueError("high must be >= low")
        return self


class PredictResponse(BaseModel):
    ticker: str
    alpha_score: float = Field(..., description="Alpha score 0-100; higher = stronger buy signal")
    return_prediction: float = Field(..., description="Predicted excess return (-0.5 to 0.5)")
    confidence: float = Field(..., description="Model confidence 0-1")
    model_version: str
    prediction_id: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    db_ok: bool


class MetricsResponse(BaseModel):
    total_predictions: int
    avg_alpha_score: float | None
    avg_confidence: float | None
    recent_sample_size: int | None = None


class DriftRequest(BaseModel):
    feature_name: str = Field(..., min_length=1)
    values: list[float] = Field(..., min_length=2)


class DriftResponse(BaseModel):
    feature_name: str
    ks_statistic: float
    p_value: float
    drift_detected: bool
