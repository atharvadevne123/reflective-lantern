"""Time-series forecasting utilities: trend decomposition and seasonal baselines."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def simple_moving_average(values: list[float], window: int) -> list[float]:
    """Compute a simple moving average over *window* periods.

    Args:
        values: Time-ordered consumption readings.
        window: Number of periods in the rolling window.

    Returns:
        Smoothed series of the same length (leading values use partial windows).
    """
    arr = np.array(values, dtype=float)
    result = np.convolve(arr, np.ones(window) / window, mode="full")[: len(arr)]
    return result.tolist()


def seasonal_baseline(values: list[float], period: int = 24) -> list[float]:
    """Compute a seasonal baseline by averaging each position modulo *period*.

    Args:
        values: Time-ordered readings (e.g. hourly).
        period: Seasonality length (default 24 for hourly data).

    Returns:
        Seasonal baseline the same length as *values*.
    """
    arr = np.array(values, dtype=float)
    n = len(arr)
    bucket_means = np.zeros(period)
    counts = np.zeros(period)
    for i, v in enumerate(arr):
        bucket_means[i % period] += v
        counts[i % period] += 1
    with np.errstate(invalid="ignore"):
        bucket_means = np.where(counts > 0, bucket_means / counts, 0.0)
    return [float(bucket_means[i % period]) for i in range(n)]


def forecast_linear_trend(values: list[float], horizon: int = 24) -> list[float]:
    """Fit a linear trend and extrapolate *horizon* steps ahead.

    Args:
        values: Historical readings.
        horizon: Number of future steps to forecast.

    Returns:
        List of *horizon* forecasted values.
    """
    arr = np.array(values, dtype=float)
    x = np.arange(len(arr))
    slope, intercept = np.polyfit(x, arr, 1)
    future_x = np.arange(len(arr), len(arr) + horizon)
    forecasted = slope * future_x + intercept
    logger.debug("Linear trend: slope=%.4f intercept=%.4f horizon=%d", slope, intercept, horizon)
    return forecasted.tolist()


def detect_spikes(values: list[float], z_threshold: float = 3.0) -> list[int]:
    """Return indices where consumption deviates more than *z_threshold* standard deviations.

    Args:
        values: Time-ordered readings.
        z_threshold: Number of standard deviations to flag as a spike.

    Returns:
        List of spike indices (empty list when std is near zero or series is empty).
    """
    if not values:
        return []
    arr = np.array(values, dtype=float)
    mean, std = float(arr.mean()), float(arr.std())
    if std < 1e-9:
        return []
    z_scores = np.abs((arr - mean) / std)
    return [int(i) for i in np.where(z_scores > z_threshold)[0]]


def peak_hours(values: list[float], top_n: int = 3) -> list[int]:
    """Return the indices of the *top_n* highest consumption readings.

    Args:
        values: Hourly consumption readings (length ≥ top_n).
        top_n: Number of peak periods to return.

    Returns:
        Indices of the highest *top_n* values, in descending order of magnitude.
    """
    if not values:
        return []
    arr = np.array(values, dtype=float)
    indices = list(np.argsort(arr)[::-1][: min(top_n, len(arr))])
    return [int(i) for i in indices]


def cumulative_sum(values: list[float]) -> list[float]:
    """Return the running cumulative sum of *values*.

    Args:
        values: Numeric series.

    Returns:
        List of the same length where element i is sum(values[:i+1]).
    """
    result: list[float] = []
    total = 0.0
    for v in values:
        total += v
        result.append(total)
    return result


def moving_max(values: list[float], window: int = 3) -> list[float]:
    """Return the rolling maximum over *window* periods.

    The first (window-1) entries are NaN-padded.

    Args:
        values: Numeric series.
        window: Rolling window size (must be >= 1).

    Returns:
        List of rolling max values, NaN-padded at the start.
    """
    if not values:
        return []
    if len(values) < window:
        return [float("nan")] * len(values)
    pad = [float("nan")] * (window - 1)
    result = [max(values[i : i + window]) for i in range(len(values) - window + 1)]
    return pad + result


def normalize_series(values: list[float]) -> list[float]:
    """Scale *values* to the [0, 1] range using min-max normalization.

    Returns all-zeros when the range is zero (constant series).

    Args:
        values: Numeric series.

    Returns:
        Min-max normalized list (same length as input).
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def daily_totals(values: list[float], period: int = 24) -> list[float]:
    """Aggregate *values* into non-overlapping blocks of size *period*.

    The last block is included even if shorter than *period*.

    Args:
        values: Flat series of observations.
        period: Aggregation block size (default 24 for hourly data).

    Returns:
        List of block sums.
    """
    if not values:
        return []
    return [sum(values[i : i + period]) for i in range(0, len(values), period)]


def first_nonzero(values: list[float]) -> int:
    """Return the index of the first non-zero element.

    Args:
        values: Numeric series.

    Returns:
        Index of first non-zero value, or -1 if all zeros or list is empty.
    """
    for i, v in enumerate(values):
        if v != 0.0:
            return i
    return -1


__all__ = [
    "simple_moving_average",
    "seasonal_baseline",
    "forecast_linear_trend",
    "detect_spikes",
    "peak_hours",
    "cumulative_sum",
    "moving_max",
    "normalize_series",
    "daily_totals",
    "first_nonzero",
]
