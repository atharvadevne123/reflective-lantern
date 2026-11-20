"""Time-series trend and performance forecasting for Sport-Cast."""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def linear_trend(values: list[float]) -> dict:
    """Fit a linear trend to a sequence of values.

    Args:
        values: Time-ordered sequence of numeric observations.

    Returns:
        Dict with 'slope', 'intercept', 'next_predicted' (one step ahead).
    """
    if len(values) < 2:
        return {"slope": 0.0, "intercept": float(values[0]) if values else 0.0, "next_predicted": float(values[0]) if values else 0.0}
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    next_pred = slope * len(values) + intercept
    return {"slope": round(slope, 4), "intercept": round(intercept, 4), "next_predicted": round(next_pred, 4)}


def rolling_average(values: list[float], window: int = 5) -> list[float]:
    """Compute a simple moving average with the given window.

    Args:
        values: Input time series.
        window: Rolling window size.

    Returns:
        List of rolling averages (shorter than input for the first window elements).
    """
    if not values:
        return []
    arr = np.array(values, dtype=float)
    result = []
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        result.append(round(float(arr[start:i + 1].mean()), 4))
    return result


def detect_form_trend(win_rates: list[float]) -> str:
    """Classify a team's recent form trend from win-rate history.

    Args:
        win_rates: Ordered list of per-match win-rate values (0–1).

    Returns:
        One of 'improving', 'declining', or 'stable'.
    """
    if len(win_rates) < 3:
        return "stable"
    trend = linear_trend(win_rates)
    if trend["slope"] > 0.05:
        return "improving"
    if trend["slope"] < -0.05:
        return "declining"
    return "stable"
