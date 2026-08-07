"""Peak demand prediction: identify highest-consumption hours in a forecast window."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def find_peak_hours(
    hourly_forecasts: list[float],
    top_n: int = 3,
) -> dict[str, object]:
    """Return the top-N peak consumption hours from a 24h forecast window.

    Args:
        hourly_forecasts: List of hourly energy consumption forecasts (kWh).
        top_n: Number of peak hours to return (default 3).

    Returns:
        Dict with 'peak_hours' (indices), 'peak_values', 'avg_kwh',
        'total_kwh', and 'peak_to_avg_ratio'.  Returns minimal dict
        when *hourly_forecasts* is empty.
    """
    if not hourly_forecasts:
        return {"peak_hours": [], "peak_values": [], "avg_kwh": 0.0, "total_kwh": 0.0}
    arr = np.array(hourly_forecasts, dtype=float)
    n = min(top_n, len(arr))
    indices = np.argsort(arr)[-n:][::-1].tolist()
    avg = float(arr.mean())
    logger.debug("find_peak_hours: top_%d peaks from %d readings, avg=%.2f", n, len(arr), avg)
    return {
        "peak_hours": [int(i) for i in indices],
        "peak_values": [round(float(arr[i]), 4) for i in indices],
        "avg_kwh": round(avg, 4),
        "total_kwh": round(float(arr.sum()), 4),
        "peak_to_avg_ratio": round(float(arr[indices[0]] / max(avg, 1e-6)), 4),
    }


def estimate_peak_shaving_savings(
    hourly_forecasts: list[float],
    shave_pct: float = 0.15,
    cost_per_kwh: float = 0.12,
) -> dict[str, float]:
    """Estimate cost savings from peak shaving the top consumption hours.

    Args:
        hourly_forecasts: List of hourly energy consumption forecasts (kWh).
        shave_pct: Fraction of the distribution to shave (default 15%).
        cost_per_kwh: Energy cost in USD per kWh (default $0.12).

    Returns:
        Dict with 'savings_kwh', 'savings_cost', 'shave_pct', and
        'peak_threshold_kwh'.
    """
    arr = np.array(hourly_forecasts, dtype=float)
    peak_threshold = float(np.percentile(arr, (1 - shave_pct) * 100))
    shaved = np.where(arr > peak_threshold, peak_threshold, arr)
    savings_kwh = float((arr - shaved).sum())
    return {
        "savings_kwh": round(savings_kwh, 4),
        "savings_cost": round(savings_kwh * cost_per_kwh, 4),
        "shave_pct": shave_pct,
        "peak_threshold_kwh": round(peak_threshold, 4),
    }


def demand_flexibility_score(
    hourly_forecasts: list[float],
    flexible_hours: list[int] | None = None,
) -> dict[str, object]:
    """Score how much demand can be shifted to off-peak hours.

    Computes the fraction of total energy that falls in flexible hours
    (default: 22:00–06:00, i.e. hours 22–23 and 0–6).

    Args:
        hourly_forecasts: List of exactly 24 hourly consumption values.
        flexible_hours: Hour indices considered flexible. Defaults to
            off-peak hours [22, 23, 0, 1, 2, 3, 4, 5, 6].

    Returns:
        Dict with 'flexible_kwh', 'peak_kwh', 'flexibility_ratio' (0–1),
        and 'shiftable_pct' as percentage.
    """
    if flexible_hours is None:
        flexible_hours = [22, 23, 0, 1, 2, 3, 4, 5, 6]
    arr = np.array(hourly_forecasts, dtype=float)
    total = float(arr.sum())
    flex_mask = np.zeros(len(arr), dtype=bool)
    for h in flexible_hours:
        if 0 <= h < len(arr):
            flex_mask[h] = True
    flexible_kwh = float(arr[flex_mask].sum())
    peak_kwh = total - flexible_kwh
    ratio = flexible_kwh / max(total, 1e-6)
    return {
        "flexible_kwh": round(flexible_kwh, 4),
        "peak_kwh": round(peak_kwh, 4),
        "flexibility_ratio": round(ratio, 4),
        "shiftable_pct": round(ratio * 100.0, 2),
    }


__all__ = [
    "demand_flexibility_score",
    "estimate_peak_shaving_savings",
    "find_peak_hours",
]
