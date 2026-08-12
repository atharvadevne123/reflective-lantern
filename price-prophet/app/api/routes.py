"""
API routes for Price-Prophet.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas import PriceRequest, PriceResponse
from app.config import settings

router = APIRouter()


@router.get("/health")
def health() -> dict:
    """Health-check endpoint."""
    return {"status": "ok", "app": settings.app_name}


@router.post("/price", response_model=PriceResponse)
def get_optimal_price(request: PriceRequest) -> PriceResponse:
    """Return a rule-based optimal price when no model is loaded."""
    if request.base_price <= 0:
        raise HTTPException(status_code=422, detail="base_price must be positive.")

    min_price = request.base_price * settings.min_price_multiplier
    max_price = request.base_price * settings.max_price_multiplier
    optimal = max(min_price, min(max_price, request.base_price))
    revenue_est = round(optimal * max(request.demand, 0.0), 2)

    return PriceResponse(
        product_id=request.product_id,
        optimal_price=round(optimal, 2),
        revenue_estimate=revenue_est,
        strategy="rule_based",
        confidence=0.5,
    )
