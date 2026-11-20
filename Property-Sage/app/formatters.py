"""Response formatting utilities for Property-Sage API."""

from __future__ import annotations

from typing import Any


def format_prediction_response(
    prediction: dict[str, float],
    request_id: str,
    neighborhood: str,
) -> dict[str, Any]:
    """Enrich a raw prediction dict with metadata for API responses.

    Args:
        prediction: Raw prediction dict from model.predict().
        request_id: UUID string for this request.
        neighborhood: Neighbourhood name from the input.

    Returns:
        Enriched response dict with formatted strings and metadata.
    """
    from app.utils import format_currency
    return {
        "request_id": request_id,
        "neighborhood": neighborhood,
        "predicted_price": prediction.get("predicted_price", 0.0),
        "predicted_price_formatted": format_currency(prediction.get("predicted_price", 0.0)),
        "predicted_rental_yield": prediction.get("predicted_rental_yield", 0.0),
        "predicted_rental_yield_pct": f"{prediction.get('predicted_rental_yield', 0.0) * 100:.2f}%",
        "estimated_annual_rental": prediction.get("estimated_annual_rental", 0.0),
        "estimated_annual_rental_formatted": format_currency(prediction.get("estimated_annual_rental", 0.0)),
        "estimated_monthly_rental": prediction.get("estimated_monthly_rental", 0.0),
        "estimated_monthly_rental_formatted": format_currency(prediction.get("estimated_monthly_rental", 0.0)),
    }


def format_metrics_response(metrics: dict[str, Any]) -> dict[str, Any]:
    """Round and label model metrics for human-readable API output.

    Args:
        metrics: Raw metrics dict from model.get_metrics().

    Returns:
        Formatted metrics dict with rounded values.
    """
    if not metrics:
        return {"status": "no_metrics_available"}
    return {
        "price_model": {
            "r2_mean": round(metrics.get("price_r2_mean", 0.0), 4),
            "r2_std": round(metrics.get("price_r2_std", 0.0), 4),
            "rmse": round(metrics.get("price_rmse", 0.0), 2),
            "mae": round(metrics.get("price_mae", 0.0), 2),
        },
        "rental_model": {
            "r2_mean": round(metrics.get("rental_r2_mean", 0.0), 4),
            "r2_std": round(metrics.get("rental_r2_std", 0.0), 4),
            "rmse": round(metrics.get("rental_rmse", 0.0), 4),
            "mae": round(metrics.get("rental_mae", 0.0), 4),
        },
        "n_train": metrics.get("n_train", 0),
    }
