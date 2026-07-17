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


def cumulative_consumption(values: list[float]) -> list[float]:
    """Return a cumulative sum of consumption values.

    Args:
        values: List of periodic (e.g. hourly) consumption readings in kWh.

    Returns:
        List of the same length where element i is sum(values[:i+1]).
    """
    if not values:
        return []
    result: list[float] = []
    total = 0.0
    for v in values:
        total += v
        result.append(round(total, 6))
    return result


def exponential_moving_average(values: list[float], alpha: float = 0.3) -> list[float]:
    """Compute an exponential moving average with smoothing factor *alpha*.

    Args:
        values: Time-ordered consumption readings.
        alpha: Smoothing factor in (0, 1]; higher values weight recent obs more.

    Returns:
        EMA series of the same length as *values*.

    Raises:
        ValueError: If *alpha* is not in (0, 1].
    """
    if not (0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if not values:
        return []
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return [round(x, 6) for x in result]


def resample_hourly_to_daily(hourly: list[float]) -> list[float]:
    """Aggregate hourly readings into daily totals.

    Args:
        hourly: List of hourly consumption values; length need not be a multiple of 24.

    Returns:
        List of daily totals; the last partial day is included if it exists.
    """
    if not hourly:
        return []
    daily: list[float] = []
    for i in range(0, len(hourly), 24):
        chunk = hourly[i : i + 24]
        daily.append(round(sum(chunk), 6))
    return daily


def forecast_trend_with_seasonality(
    values: list[float],
    horizon: int = 24,
    period: int = 24,
) -> list[float]:
    """Forecast by combining a linear trend with a seasonal baseline.

    Decomposes the series into trend + seasonal components and extrapolates both.

    Args:
        values: Historical readings (at least 2 * period elements recommended).
        horizon: Number of future steps to forecast.
        period: Seasonal period length (default 24 for hourly data).

    Returns:
        List of *horizon* forecasted values (clipped to 0 for energy context).
    """
    if not values or horizon < 1:
        return []
    arr = np.array(values, dtype=float)
    n = len(arr)
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, arr, 1)
    trend = slope * x + intercept
    residual = arr - trend
    bucket_means = np.zeros(period)
    counts = np.zeros(period)
    for i, v in enumerate(residual):
        bucket_means[i % period] += v
        counts[i % period] += 1
    with np.errstate(invalid="ignore"):
        bucket_means = np.where(counts > 0, bucket_means / counts, 0.0)
    future_x = np.arange(n, n + horizon, dtype=float)
    future_trend = slope * future_x + intercept
    future_seasonal = np.array([bucket_means[i % period] for i in range(n, n + horizon)])
    forecast = future_trend + future_seasonal
    return np.clip(forecast, 0, None).tolist()


def moving_range(values: list[float]) -> list[float]:
    """Compute the moving range (absolute successive differences) of a series.

    The moving range is commonly used in statistical process control to
    estimate short-term process variation.

    Args:
        values: Ordered list of numeric observations.

    Returns:
        List of absolute differences between consecutive values (length = len-1),
        or an empty list if *values* has fewer than 2 elements.
    """
    if len(values) < 2:
        return []
    return [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]


def consumption_variance(values: list[float]) -> float:
    """Return the population variance of an energy consumption series.

    Args:
        values: List of kWh readings.

    Returns:
        Population variance as a float, or 0.0 for a series of length < 2.
    """
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def detect_plateau(values: list[float], tolerance: float = 0.5) -> list[tuple[int, int]]:
    """Detect flat plateaus in a time series where variance is near zero.

    Returns a list of (start, end) index tuples for runs where consecutive
    values differ by at most *tolerance*.
    """
    if len(values) < 2:
        return []
    plateaus: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(values)):
        if abs(values[i] - values[i - 1]) > tolerance:
            if i - start >= 2:
                plateaus.append((start, i - 1))
            start = i
    if len(values) - start >= 2:
        plateaus.append((start, len(values) - 1))
    return plateaus


def clip_outliers(values: list[float], lower_pct: float = 5.0, upper_pct: float = 95.0) -> list[float]:
    """Clip series values to the given percentile bounds.

    Args:
        values: Input readings.
        lower_pct: Lower bound percentile (0–100).
        upper_pct: Upper bound percentile (0–100).

    Returns:
        Series with values clipped to the [lower_pct, upper_pct] range.
    """
    if not values:
        return []
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def percentile(p: float) -> float:
        idx = (p / 100) * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)

    lo = percentile(lower_pct)
    hi = percentile(upper_pct)
    return [max(lo, min(hi, v)) for v in values]
