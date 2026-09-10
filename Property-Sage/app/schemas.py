"""Shared Pydantic response schemas for the Property-Sage API.

Centralising schemas here keeps main.py concise and allows
them to be imported in tests without importing the full FastAPI app.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Standard liveness response."""

    status: str = Field(..., examples=["ok"])
    service: str = Field(..., examples=["property-sage"])
    version: str = Field(..., examples=["1.0.0"])


class PredictionResponse(BaseModel):
    """Output schema for a successful property valuation."""

    request_id: str = Field(..., description="UUID for this inference request.")
    predicted_price: float = Field(
        ..., description="Estimated market value in USD.", examples=[452300.50]
    )
    predicted_rental_yield: float = Field(
        ..., description="Gross rental yield (0–1).", examples=[0.0521]
    )
    estimated_annual_rental: float = Field(
        ..., description="Annual rental income in USD.", examples=[23565.0]
    )
    estimated_monthly_rental: float = Field(
        ..., description="Monthly rental income in USD.", examples=[1963.75]
    )
    neighborhood: str = Field(..., examples=["suburb"])
    property_type: str = Field(..., examples=["house"])


class DriftCheckResponse(BaseModel):
    """Drift detection result for a monitoring window."""

    status: str
    drift_detected: bool | None = None
    checks: list[dict] = Field(default_factory=list)


class MetricsResponse(BaseModel):
    """Combined model performance and prediction statistics."""

    model_performance: dict = Field(default_factory=dict)
    prediction_stats: dict = Field(default_factory=dict)


class NeighbourhoodReportResponse(BaseModel):
    """RAG-retrieved market intelligence for one neighbourhood."""

    neighbourhood: str
    market_report: str


class NeighbourhoodSearchResponse(BaseModel):
    """Semantic search results over the neighbourhood knowledge base."""

    query: str
    results: list[dict]
