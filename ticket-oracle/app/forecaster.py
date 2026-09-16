"""Time-series ticket volume forecasting for Ticket-Oracle.

Uses simple moving average (SMA) and linear trend to project ticket
intake volume over the next N hours, helping agents plan capacity.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_sma(values: list[float], window: int) -> list[float]:
    """Compute a simple moving average over a list of values.

    Args:
        values: Time-ordered list of numeric observations.
        window: Rolling window size.

    Returns:
        List of SMA values (length = len(values) - window + 1).

    Raises:
        ValueError: If window is larger than the input length.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if window > len(values):
        raise ValueError(f"window ({window}) exceeds series length ({len(values)})")
    arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid").tolist()


def fit_linear_trend(values: list[float]) -> tuple[float, float]:
    """Fit a linear trend y = slope * t + intercept via OLS.

    Args:
        values: Equally-spaced time-series observations.

    Returns:
        Tuple of (slope, intercept).
    """
    n = len(values)
    if n < 2:
        return 0.0, float(values[0]) if values else 0.0
    t = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(t, values, 1)
    return float(slope), float(intercept)


def forecast_volume(
    historical_hourly_counts: list[float],
    horizon_hours: int = 24,
) -> dict[str, Any]:
    """Forecast ticket intake volume for the next horizon_hours hours.

    Combines a linear trend extrapolation with the last known SMA value
    to produce a simple but robust forecast.

    Args:
        historical_hourly_counts: Past hourly ticket counts (oldest first).
        horizon_hours: Number of hours to forecast forward.

    Returns:
        Dictionary with forecasted values and trend metadata.
    """
    if len(historical_hourly_counts) < 4:
        return {
            "forecast": [float(historical_hourly_counts[-1])] * horizon_hours if historical_hourly_counts else [],
            "trend": "insufficient_data",
            "slope": 0.0,
        }

    slope, intercept = fit_linear_trend(historical_hourly_counts)
    n = len(historical_hourly_counts)

    forecasted = []
    for h in range(1, horizon_hours + 1):
        point = slope * (n + h) + intercept
        forecasted.append(max(0.0, round(point, 2)))

    trend_label = "increasing" if slope > 0.5 else "decreasing" if slope < -0.5 else "stable"

    logger.info(
        "volume_forecast_computed",
        extra={"slope": round(slope, 4), "trend": trend_label, "horizon_hours": horizon_hours},
    )

    return {
        "forecast": forecasted,
        "trend": trend_label,
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "n_historical": n,
    }


def detect_volume_spike(
    hourly_counts: list[float],
    z_threshold: float = 2.5,
) -> dict[str, Any]:
    """Detect anomalous ticket intake spikes using z-score.

    Args:
        hourly_counts: Recent hourly ticket counts.
        z_threshold: Z-score cutoff for spike labelling.

    Returns:
        Dictionary flagging each hour as spike/normal.
    """
    if len(hourly_counts) < 5:
        return {"spikes": [], "mean": 0.0, "std": 0.0}

    arr = np.array(hourly_counts, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std())

    if std < 1e-6:
        return {"spikes": [], "mean": round(mean, 2), "std": 0.0}

    z_scores = ((arr - mean) / std).tolist()
    spikes = [
        {"hour_index": i, "count": hourly_counts[i], "z_score": round(z, 3)}
        for i, z in enumerate(z_scores)
        if abs(z) >= z_threshold
    ]

    return {"spikes": spikes, "mean": round(mean, 2), "std": round(std, 2)}
