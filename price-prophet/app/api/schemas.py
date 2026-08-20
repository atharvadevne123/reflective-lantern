"""
Pydantic request and response schemas for the Price-Prophet API.

All schemas use Pydantic v2 where available and fall back gracefully to
v1 field validators via the shared ``validator`` decorator shim.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PriceRequest(BaseModel):
    """Incoming pricing request for a single product.

    Attributes
    ----------
    product_id:
        Unique identifier for the product.
    base_price:
        Current selling price of the product.
    demand:
        Observed or forecasted unit demand at *base_price*.
    competition_price:
        Best competing offer price (0.0 if unknown).
    day_of_week:
        Day index 0 (Monday) - 6 (Sunday).
    is_weekend:
        Whether the observation falls on a weekend.
    category:
        Product category label (used for segment-level features).
    """

    product_id: str
    base_price: float = Field(..., gt=0, description="Current price; must be positive.")
    demand: float = Field(..., ge=0, description="Observed demand; must be non-negative.")
    competition_price: float = Field(default=0.0, ge=0)
    day_of_week: int = Field(default=0, ge=0, le=6)
    is_weekend: bool = False
    category: str = "default"


class PriceResponse(BaseModel):
    """Optimised pricing recommendation for a single product.

    Attributes
    ----------
    product_id:
        Echo of the incoming *product_id*.
    optimal_price:
        Recommended price for the next selling period.
    revenue_estimate:
        Projected revenue at *optimal_price*.
    strategy:
        Name of the pricing strategy applied.
    confidence:
        Model confidence score in [0, 1].
    """

    product_id: str
    optimal_price: float
    revenue_estimate: float
    strategy: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class ModelMetrics(BaseModel):
    """Summary metrics for a trained model.

    Attributes
    ----------
    mae:
        Mean absolute error on the evaluation set.
    rmse:
        Root mean squared error.
    r_squared:
        Coefficient of determination.
    n_samples:
        Number of samples used for evaluation.
    """

    mae: float
    rmse: float
    r_squared: float
    n_samples: int = Field(..., ge=0)


class HealthResponse(BaseModel):
    """Health-check response.

    Attributes
    ----------
    status:
        Human-readable service status (e.g. ``"ok"``).
    version:
        API version string.
    is_model_loaded:
        Whether a pricing model is currently loaded and ready for
        inference.
    """

    model_config = ConfigDict(protected_namespaces=())

    status: str
    version: str
    is_model_loaded: bool = False


class BatchPriceRequest(BaseModel):
    """Batch pricing request for multiple products.

    Attributes
    ----------
    items:
        List of individual pricing requests; maximum 100 per call.
    """

    items: list[PriceRequest] = Field(..., min_length=1, max_length=100)


class BatchPriceResponse(BaseModel):
    """Batch pricing response with one result per request item.

    Attributes
    ----------
    results:
        Ordered list of pricing responses matching *items* in the request.
    count:
        Total number of products priced in this batch.
    """

    results: list[PriceResponse]
    count: int = Field(..., ge=0)
