"""Property valuation report generation for Property-Sage."""

from __future__ import annotations

from typing import Any


def property_valuation_report(
    input_data: dict[str, Any],
    prediction: dict[str, float],
    drift_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a structured valuation report for a single property.

    Args:
        input_data: Raw property attributes from the request.
        prediction: Model output dict with price and yield predictions.
        drift_result: Optional drift detection result to include.

    Returns:
        Structured report dict suitable for API response or PDF generation.
    """
    from app.features import compute_cap_rate, compute_price_per_sqft, property_age_category
    from app.monitoring import yield_category
    from app.utils import format_currency

    price = prediction.get("predicted_price", 0.0)
    rental_yield = prediction.get("predicted_rental_yield", 0.0)
    annual_rental = prediction.get("estimated_annual_rental", price * rental_yield)
    sqft = float(input_data.get("sqft", 0))
    year_built = int(input_data.get("year_built", 2000))

    return {
        "summary": {
            "predicted_price": price,
            "predicted_price_formatted": format_currency(price),
            "rental_yield_pct": f"{rental_yield * 100:.2f}%",
            "yield_category": yield_category(rental_yield),
            "price_per_sqft": compute_price_per_sqft(price, sqft),
            "cap_rate": compute_cap_rate(annual_rental, price),
            "property_age_category": property_age_category(year_built),
        },
        "input": input_data,
        "prediction": prediction,
        "drift": drift_result or {"drift_detected": False},
    }
