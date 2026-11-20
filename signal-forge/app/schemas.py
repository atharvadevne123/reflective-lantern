"""Pydantic v2 request/response schemas with full OpenAPI descriptions."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Input payload for the /predict endpoint."""

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Asset ticker symbol (e.g. AAPL, BTC-USD)",
        examples=["AAPL"],
        json_schema_extra={"example": "AAPL"},
    )
    close: float = Field(
        ...,
        gt=0,
        description="Latest closing price in USD",
        examples=[195.0],
    )
    volume: float = Field(
        ...,
        ge=0,
        description="Latest traded volume (shares or contracts)",
        examples=[80_000_000.0],
    )
    market_return: float = Field(
        0.0,
        description="Benchmark (e.g. S&P 500) daily return fraction. Defaults to 0.",
        examples=[0.005],
    )

    @field_validator("ticker")
    @classmethod
    def normalise_ticker(cls, v: str) -> str:
        """Upper-case and strip whitespace from the ticker."""
        return v.upper().strip()

    model_config = {
        "json_schema_extra": {
            "example": {
                "ticker": "AAPL",
                "close": 195.0,
                "volume": 80_000_000.0,
                "market_return": 0.005,
            }
        }
    }


class SimilarPeriod(BaseModel):
    """A single FAISS nearest-neighbour result."""

    id: int | None = None
    ticker: str | None = None
    regime: str | None = None
    date: str | None = None
    distance: float


class PredictResponse(BaseModel):
    """Regime classification and risk score output."""

    ticker: str
    regime: str = Field(..., description="Detected market regime: bull/bear/sideways/volatile")
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Portfolio risk score (0=low, 1=high)")
    volatility: float
    momentum: float
    correlation: float
    volume_ratio: float
    beta: float
    similar_periods: list[dict[str, Any]] = Field(default_factory=list)
    model_version: str


class HealthResponse(BaseModel):
    """Health check output."""

    status: str
    version: str


class MetricsResponse(BaseModel):
    """Runtime metrics output."""

    request_count: int
    uptime_seconds: float
    version: str
    drift: dict[str, Any]


class DriftCheckResponse(BaseModel):
    """Drift detection output."""

    drift_detected: bool
    features: list[dict[str, Any]]
