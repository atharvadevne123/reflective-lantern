"""Time-series load trend analysis and seasonal decomposition."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_trend(
    loads: list[float],
    window: int = 24,
) -> dict:
    """Compute simple moving average trend and slope for a load series."""
    if len(loads) < window:
        return {"trend": None, "slope": None, "direction": "unknown"}

    arr = np.array(loads, dtype=float)
    sma = np.convolve(arr, np.ones(window) / window, mode="valid")

    if len(sma) >= 2:
        slope = float(np.polyfit(np.arange(len(sma)), sma, 1)[0])
        direction = "increasing" if slope > 10 else "decreasing" if slope < -10 else "stable"
    else:
        slope = 0.0
        direction = "stable"

    return {
        "sma_last": round(float(sma[-1]), 2) if len(sma) > 0 else None,
        "slope_mw_per_hour": round(slope, 3),
        "direction": direction,
        "window_size": window,
        "series_length": len(loads),
    }


def detect_load_spikes(
    loads: list[float],
    z_threshold: float = 3.0,
) -> list[dict]:
    """Detect load spikes using Z-score method."""
    if len(loads) < 10:
        return []

    arr = np.array(loads, dtype=float)
    mean = arr.mean()
    std = arr.std()
    if std == 0:
        return []

    z_scores = np.abs((arr - mean) / std)
    spikes = [
        {
            "index": int(i),
            "load_mw": round(float(arr[i]), 2),
            "z_score": round(float(z_scores[i]), 3),
            "deviation_mw": round(float(arr[i] - mean), 2),
        }
        for i in np.where(z_scores > z_threshold)[0]
    ]
    return spikes


def seasonal_summary(
    loads: list[float],
    period: int = 24,
) -> dict:
    """Compute per-period mean to approximate seasonal pattern."""
    if len(loads) < period:
        return {"period": period, "seasonal_means": []}

    arr = np.array(loads, dtype=float)
    n_complete = (len(arr) // period) * period
    matrix = arr[:n_complete].reshape(-1, period)
    seasonal_means = matrix.mean(axis=0).tolist()

    return {
        "period": period,
        "seasonal_means": [round(v, 2) for v in seasonal_means],
        "peak_period_index": int(np.argmax(seasonal_means)),
        "trough_period_index": int(np.argmin(seasonal_means)),
    }
