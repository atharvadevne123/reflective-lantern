"""
API routes for Price-Prophet.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.api.schemas import BatchPriceRequest, BatchPriceResponse, PriceRequest, PriceResponse
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health() -> dict:
    """Health-check endpoint."""
    return {"status": "ok", "app": settings.app_name, "version": settings.version}


@router.get("/version")
def version() -> dict:
    """Return the application version."""
    return {"version": settings.version, "app": settings.app_name}


@router.get("/metrics")
def metrics() -> dict:
    """Return basic operational metrics."""
    return {
        "status": "ok",
        "cache_ttl_seconds": settings.cache_ttl_seconds,
        "max_price_multiplier": settings.max_price_multiplier,
        "min_price_multiplier": settings.min_price_multiplier,
    }


@router.post("/price", response_model=PriceResponse)
def get_optimal_price(request: PriceRequest) -> PriceResponse:
    """Return a rule-based optimal price when no model is loaded."""
    if request.base_price <= 0:
        raise HTTPException(status_code=422, detail="base_price must be positive.")

    min_price = request.base_price * settings.min_price_multiplier
    max_price = request.base_price * settings.max_price_multiplier
    optimal = max(min_price, min(max_price, request.base_price))
    revenue_est = round(optimal * max(request.demand, 0.0), 2)

    logger.info(
        "price request product_id=%s base_price=%.2f optimal=%.2f",
        request.product_id,
        request.base_price,
        optimal,
    )

    return PriceResponse(
        product_id=request.product_id,
        optimal_price=round(optimal, 2),
        revenue_estimate=revenue_est,
        strategy="rule_based",
        confidence=0.5,
    )


@router.post("/price/batch", response_model=BatchPriceResponse)
def get_batch_optimal_prices(request: BatchPriceRequest) -> BatchPriceResponse:
    """Return rule-based optimal prices for multiple products in one call."""
    results: list[PriceResponse] = []
    for item in request.items:
        min_price = item.base_price * settings.min_price_multiplier
        max_price = item.base_price * settings.max_price_multiplier
        optimal = max(min_price, min(max_price, item.base_price))
        revenue_est = round(optimal * max(item.demand, 0.0), 2)
        results.append(
            PriceResponse(
                product_id=item.product_id,
                optimal_price=round(optimal, 2),
                revenue_estimate=revenue_est,
                strategy="rule_based",
                confidence=0.5,
            )
        )
    logger.info("batch price request items=%d", len(results))
    return BatchPriceResponse(results=results, count=len(results))
