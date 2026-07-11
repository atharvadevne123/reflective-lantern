"""Pydantic request/response schemas for Realty-Edge API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

_MAX_SQFT = 50_000
_MAX_BEDROOMS = 20
_MAX_BATHROOMS = 20.0
_MIN_YEAR = 1800
_MAX_YEAR = 2026
_BATCH_MAX = 100
_MAX_TOP_K = 20


class PropertyInput(BaseModel):
    """Input schema for a single property valuation request.

    All numeric fields are validated at the boundary; ``renovation_year``
    must not precede ``year_built`` when both are supplied.
    """

    sqft: float = Field(..., gt=0, le=_MAX_SQFT, description="Total living area in square feet")
    bedrooms: int = Field(..., ge=1, le=_MAX_BEDROOMS)
    bathrooms: float = Field(..., ge=0.5, le=_MAX_BATHROOMS)
    lot_size: float = Field(0.0, ge=0, description="Lot size in square feet")
    year_built: int = Field(..., ge=_MIN_YEAR, le=_MAX_YEAR)
    renovation_year: int | None = Field(None, ge=_MIN_YEAR, le=_MAX_YEAR)
    condition_score: float = Field(5.0, ge=1.0, le=10.0)
    zipcode: str = Field(..., min_length=5, max_length=10)
    city: str = Field("", max_length=100)
    state: str = Field("", max_length=50)
    school_score: float = Field(5.0, ge=0.0, le=10.0)
    transit_score: float = Field(5.0, ge=0.0, le=10.0)
    walkability_score: float = Field(5.0, ge=0.0, le=10.0)
    crime_rate: float = Field(0.3, ge=0.0, le=1.0)
    median_neighborhood_price: float = Field(300_000.0, gt=0)
    median_price_per_sqft: float = Field(200.0, gt=0)
    avg_rental_yield: float = Field(0.06, ge=0.0, le=1.0)
    listing_days: int = Field(30, ge=0, le=3650)
    list_price: float | None = Field(None, gt=0)

    @field_validator("renovation_year")
    @classmethod
    def renovation_after_built(cls, v: int | None, info: object) -> int | None:
        if (
            v is not None
            and hasattr(info, "data")
            and "year_built" in info.data
            and v < info.data["year_built"]
        ):
            raise ValueError("renovation_year must be >= year_built")
        return v


class PredictionResponse(BaseModel):
    """Response schema for a single property valuation prediction."""

    predicted_value: float
    investment_score: float
    confidence_band_low: float
    confidence_band_high: float
    model_version: str
    correlation_id: str


class BatchPropertyInput(BaseModel):
    """Input schema for a batch of up to 100 property valuation requests."""

    properties: list[PropertyInput] = Field(..., min_length=1, max_length=_BATCH_MAX)


class BatchPredictionResponse(BaseModel):
    """Response schema wrapping predictions for a batch request."""

    predictions: list[PredictionResponse]
    count: int


class ComparableRequest(BaseModel):
    """Request schema for comparable-property search via FAISS."""

    property: PropertyInput
    top_k: int = Field(5, ge=1, le=_MAX_TOP_K)


class ComparableResponse(BaseModel):
    """Response schema containing the nearest comparable properties."""

    comparables: list[dict]
    query_vector_dim: int


class NeighborhoodStatsResponse(BaseModel):
    """Response schema for aggregated neighbourhood statistics."""

    zipcode: str
    median_price: float
    median_price_per_sqft: float
    school_score: float
    transit_score: float
    walkability_score: float
    crime_rate: float
    avg_rental_yield: float


class DriftStatusResponse(BaseModel):
    """Response schema for the /drift endpoint."""

    drift_reports: list[dict]
    total_predictions: int


class HealthResponse(BaseModel):
    """Response schema for the /health liveness check."""

    status: str
    model_version: str
    db_connected: bool


class MetricsResponse(BaseModel):
    """Response schema for model training metrics."""

    r2_mean: float | None = None
    rmse_mean: float | None = None
    n_features: int | None = None
    n_samples: int | None = None
    model_version: str
    note: str | None = None
