"""Neighbourhood and portfolio statistics helpers for Property-Sage."""

from __future__ import annotations

import statistics
from typing import Any


def neighborhood_price_stats(prices: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of property prices.

    Args:
        prices: List of numeric price values.

    Returns:
        Dict with mean, median, std_dev, min, max, and count.
    """
    if not prices:
        return {"mean": 0.0, "median": 0.0, "std_dev": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    return {
        "mean": round(statistics.mean(prices), 2),
        "median": round(statistics.median(prices), 2),
        "std_dev": round(statistics.stdev(prices) if len(prices) > 1 else 0.0, 2),
        "min": round(min(prices), 2),
        "max": round(max(prices), 2),
        "count": len(prices),
    }


def neighborhood_benchmark(
    subject_price: float,
    neighborhood_prices: list[float],
) -> dict[str, Any]:
    """Benchmark a subject property against its neighbourhood.

    Args:
        subject_price: Price of the property to benchmark.
        neighborhood_prices: List of comparable prices in the area.

    Returns:
        Dict with percentile rank, delta_from_mean, and above_median flag.
    """
    if not neighborhood_prices:
        return {"percentile_rank": 0.0, "delta_from_mean": 0.0, "above_median": False}
    below = sum(1 for p in neighborhood_prices if p < subject_price)
    percentile = round(below / len(neighborhood_prices) * 100, 1)
    mean = statistics.mean(neighborhood_prices)
    median = statistics.median(neighborhood_prices)
    return {
        "percentile_rank": percentile,
        "delta_from_mean": round(subject_price - mean, 2),
        "above_median": subject_price > median,
    }
