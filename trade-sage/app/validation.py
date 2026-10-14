"""Input validation models (Pydantic v2)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class OHLCVRow(BaseModel):
    open: float = Field(gt=0, description="Opening price")
    high: float = Field(gt=0, description="Intraday high")
    low: float = Field(gt=0, description="Intraday low")
    close: float = Field(gt=0, description="Closing price")
    volume: float = Field(ge=0, description="Trading volume")

    @model_validator(mode="after")
    def high_gte_low(self) -> OHLCVRow:
        if self.high < self.low:
            raise ValueError("high must be >= low")
        return self


class PredictRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=16, description="Equity ticker symbol")
    horizon_days: int = Field(
        default=5, ge=1, le=60, description="Forecast horizon in trading days"
    )
    ohlcv: list[OHLCVRow] = Field(
        min_length=60,
        description="At least 60 rows of OHLCV data ordered oldest-first",
    )

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        return v.upper().strip()


class PortfolioHoldingInput(BaseModel):
    ticker: str = Field(min_length=1, max_length=16)
    weight: float = Field(gt=0, le=1.0)
    ohlcv: list[OHLCVRow] = Field(min_length=60)


class PortfolioRiskRequest(BaseModel):
    holdings: list[PortfolioHoldingInput] = Field(min_length=1, max_length=30)
    confidence_level: float = Field(default=0.95, ge=0.9, le=0.99)

    @field_validator("holdings")
    @classmethod
    def weights_sum_to_one(cls, v: list[PortfolioHoldingInput]) -> list[PortfolioHoldingInput]:
        total = sum(h.weight for h in v)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Portfolio weights must sum to 1.0, got {total:.4f}")
        return v


class DriftCheckRequest(BaseModel):
    ohlcv: list[OHLCVRow] = Field(min_length=30)


class PredictResponse(BaseModel):
    ticker: str
    prediction: int = Field(description="1=bullish, 0=bearish")
    confidence: float
    horizon_days: int
    signal: str = Field(description="'BUY' or 'SELL'")


class PortfolioRiskResponse(BaseModel):
    value_at_risk: float = Field(description="VaR at specified confidence level (daily)")
    expected_shortfall: float
    portfolio_volatility: float
    sharpe_ratio: float | None
    max_drawdown: float
    correlation_avg: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    db_connected: bool
    version: str = "1.0.0"


class MetricsResponse(BaseModel):
    total_predictions: int
    drift_alerts: int
    model_auc: float | None
    uptime_seconds: float
