"""Pydantic request/response schemas for Realty-Edge API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PropertyInput(BaseModel):
    sqft: float = Field(..., gt=0, le=50_000, description="Total living area in square feet")
    bedrooms: int = Field(..., ge=1, le=20)
    bathrooms: float = Field(..., ge=0.5, le=20)
    lot_size: float = Field(0.0, ge=0, description="Lot size in square feet")
    year_built: int = Field(..., ge=1800, le=2026)
    renovation_year: int | None = Field(None, ge=1800, le=2026)
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
    predicted_value: float
    investment_score: float
    confidence_band_low: float
    confidence_band_high: float
    model_version: str
    correlation_id: str


class BatchPropertyInput(BaseModel):
    properties: list[PropertyInput] = Field(..., min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


class ComparableRequest(BaseModel):
    property: PropertyInput
    top_k: int = Field(5, ge=1, le=20)


class ComparableResponse(BaseModel):
    comparables: list[dict]
    query_vector_dim: int


class NeighborhoodStatsResponse(BaseModel):
    zipcode: str
    median_price: float
    median_price_per_sqft: float
    school_score: float
    transit_score: float
    walkability_score: float
    crime_rate: float
    avg_rental_yield: float


class DriftStatusResponse(BaseModel):
    drift_reports: list[dict]
    total_predictions: int


class HealthResponse(BaseModel):
    status: str
    model_version: str
    db_connected: bool


class MetricsResponse(BaseModel):
    r2_mean: float | None = None
    rmse_mean: float | None = None
    n_features: int | None = None
    n_samples: int | None = None
    model_version: str
    note: str | None = None
