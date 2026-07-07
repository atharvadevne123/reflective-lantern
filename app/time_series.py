"""Time-series price trend forecasting for Realty-Edge.

Implements simple moving average (SMA), linear trend, and exponential
smoothing forecasts for neighbourhood median price series. Used by the
/neighborhood-stats endpoint to surface forward-looking price indicators.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MIN_PERIODS = 3


def compute_sma(values: list[float], window: int = 3) -> list[float]:
    """Return a simple moving average series of the same length as values.

    The first (window-1) entries are filled with NaN.

    Args:
        values: Chronologically ordered price observations.
        window: Rolling window size.

    Returns:
        List of SMA values, NaN-padded at the start.
    """
    if len(values) < window:
        return [float("nan")] * len(values)
    arr = np.array(values, dtype=float)
    sma = np.convolve(arr, np.ones(window) / window, mode="valid")
    pad = [float("nan")] * (window - 1)
    return pad + sma.tolist()


def linear_trend_forecast(
    values: list[float], horizon: int = 3
) -> dict[str, Any]:
    """Fit a linear trend and extrapolate forward by horizon steps.

    Args:
        values: Chronologically ordered price observations.
        horizon: Number of future steps to forecast.

    Returns:
        Dict with keys: slope, intercept, r_squared, forecasts (list).
    """
    n = len(values)
    if n < MIN_PERIODS:
        return {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0, "forecasts": []}
    x = np.arange(n, dtype=float)
    y = np.array(values, dtype=float)
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    forecasts = [slope * (n + i) + intercept for i in range(horizon)]
    logger.debug("Linear trend: slope=%.2f r2=%.4f", slope, r_squared)
    return {
        "slope": round(slope, 2),
        "intercept": round(intercept, 2),
        "r_squared": round(r_squared, 4),
        "forecasts": [round(f, 2) for f in forecasts],
    }


def exponential_smoothing_forecast(
    values: list[float], alpha: float = 0.3, horizon: int = 3
) -> list[float]:
    """Apply single exponential smoothing and forecast horizon steps ahead.

    Args:
        values: Chronologically ordered price observations.
        alpha: Smoothing factor in (0, 1). Higher = more weight on recent obs.
        horizon: Number of future steps.

    Returns:
        List of forecast values (length == horizon).
    """
    if not values:
        return []
    smoothed = float(values[0])
    for v in values[1:]:
        smoothed = alpha * v + (1 - alpha) * smoothed
    return [round(smoothed, 2)] * horizon
