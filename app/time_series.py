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

    Args:
        values: Time-ordered consumption readings.
        window: Number of periods to include in each rolling window.

    Returns:
        Rolling maximum series of the same length as *values*.
    """
    if not values:
        return []
    result: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(max(values[start : i + 1]))
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
        lower_pct: Lower bound percentile (0-100).
        upper_pct: Upper bound percentile (0-100).

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


def find_changepoints(
    values: list[float],
    min_segment_len: int = 5,
    threshold_multiplier: float = 2.0,
) -> list[int]:
    """Identify abrupt level-shift change points in a time series.

    Uses a simple cumulative sum (CUSUM) approach to detect points where the
    running mean changes significantly.

    Args:
        values: Time-ordered readings.
        min_segment_len: Minimum samples required between change points.
        threshold_multiplier: Number of std deviations above the CUSUM variance
            needed to flag a change point.

    Returns:
        Sorted list of indices where level shifts are detected.
    """
    if len(values) < 2 * min_segment_len:
        return []
    arr = np.array(values, dtype=float)
    n = len(arr)
    mean = arr.mean()
    std = arr.std()
    if std < 1e-9:
        return []
    cusum = np.cumsum(arr - mean)
    threshold = threshold_multiplier * std * np.sqrt(n)
    changepoints: list[int] = []
    last_cp = 0
    for i in range(min_segment_len, n - min_segment_len):
        if i - last_cp < min_segment_len:
            continue
        local_max = max(abs(cusum[last_cp : i + 1]))
        if local_max > threshold:
            changepoints.append(i)
            last_cp = i
    return changepoints


def rolling_zscore(values: list[float], window: int = 24) -> list[float]:
    """Compute rolling Z-scores for anomaly detection in streaming data.

    Each position's Z-score is computed relative to the preceding *window*
    values (or all available values for early positions).

    Args:
        values: Time-ordered readings.
        window: Rolling window size.

    Returns:
        List of Z-scores the same length as *values*. First position is 0.0.
    """
    if not values:
        return []
    zscores = [0.0]
    for i in range(1, len(values)):
        segment = values[max(0, i - window) : i]
        if len(segment) < 2:
            zscores.append(0.0)
            continue
        arr = np.array(segment)
        std = float(arr.std())
        if std < 1e-9:
            zscores.append(0.0)
        else:
            zscores.append(round((values[i] - float(arr.mean())) / std, 4))
    return zscores


def load_factor(values: list[float]) -> float:
    """Compute the load factor (average load / peak load) for a series.

    The load factor is a dimensionless ratio in [0, 1] that measures how
    efficiently a system uses its peak capacity over a period.

    Args:
        values: Time-ordered load readings (all non-negative).

    Returns:
        Load factor in [0.0, 1.0]. Returns 0.0 for an empty or all-zero series.
    """
    if not values:
        return 0.0
    peak = max(values)
    if peak == 0.0:
        return 0.0
    avg = sum(values) / len(values)
    return round(avg / peak, 6)


def peak_to_valley_ratio(values: list[float]) -> float:
    """Return the ratio of peak to minimum (valley) value in the series.

    Useful for quantifying load variability. Returns 0.0 for empty or
    all-zero (no valley) series.

    Args:
        values: Time-ordered readings.

    Returns:
        Peak-to-valley ratio; 0.0 if the minimum is zero or the series is empty.
    """
    if not values:
        return 0.0
    valley = min(values)
    if valley == 0.0:
        return 0.0
    return round(max(values) / valley, 6)


__all__ = [
    "autocorrelation",
    "clip_outliers",
    "consumption_variance",
    "cumulative_consumption",
    "cumulative_sum",
    "daily_totals",
    "detect_plateau",
    "detect_spikes",
    "exponential_moving_average",
    "find_changepoints",
    "first_nonzero",
    "forecast_linear_trend",
    "forecast_trend_with_seasonality",
    "hampel_filter",
    "holt_winters_smooth",
    "load_factor",
    "logger",
    "mape",
    "min_max_scale",
    "moving_max",
    "moving_median",
    "moving_range",
    "normalize_series",
    "pair_difference",
    "peak_hours",
    "peak_to_valley_ratio",
    "percent_change",
    "resample_hourly_to_daily",
    "rolling_zscore",
    "seasonal_baseline",
    "simple_moving_average",
    "z_normalize",
]


def moving_median(values: list[float], window: int) -> list[float]:
    """Compute a rolling median over *window* periods.

    Each position uses the preceding *window* values (or all available
    for positions near the start).

    Args:
        values: Time-ordered readings.
        window: Rolling window size (must be >= 1).

    Returns:
        List of rolling max values, NaN-padded at the start.
    """
    if not values:
        return []
    result: list[float] = []
    for i in range(len(values)):
        chunk = sorted(values[max(0, i - window + 1) : i + 1])
        n = len(chunk)
        mid = n // 2
        if n % 2 == 0:
            result.append(round((chunk[mid - 1] + chunk[mid]) / 2.0, 6))
        else:
            result.append(round(chunk[mid], 6))
    return result


def autocorrelation(values: list[float], lag: int = 1) -> float:
    """Compute the autocorrelation of *values* at the given *lag*.

    Measures how strongly the series correlates with its own past values,
    useful for detecting seasonality and trend persistence.

    Args:
        values: Time-ordered observations (at least lag+2 elements).
        lag: Time lag in number of periods (default 1 = one-step autocorrelation).

    Returns:
        Pearson autocorrelation coefficient in [-1, 1].
        Returns 0.0 when the series has insufficient length or zero variance.

    Raises:
        ValueError: If *lag* is less than 1 or *values* is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    n = len(values)
    if n <= lag + 1:
        return 0.0
    arr = np.array(values, dtype=float)
    mean = arr.mean()
    variance = float(((arr - mean) ** 2).mean())
    if variance < 1e-12:
        return 0.0
    cov = float(((arr[lag:] - mean) * (arr[:-lag] - mean)).mean())
    return round(cov / variance, 6)


def holt_winters_smooth(
    values: list[float],
    alpha: float = 0.3,
    beta: float = 0.1,
) -> list[float]:
    """Apply Holt's double exponential smoothing (trend-corrected EMA).

    Suitable for series with a linear trend but no seasonality.

    Args:
        values: Time-ordered observations (at least 2 elements).
        alpha: Level smoothing factor in (0, 1].
        beta: Trend smoothing factor in (0, 1].

    Returns:
        Smoothed series of the same length as *values*.

    Raises:
        ValueError: If *values* has fewer than 2 elements or parameters are invalid.
    """
    if len(values) < 2:
        raise ValueError("values must have at least 2 elements")
    if not (0 < alpha <= 1) or not (0 < beta <= 1):
        raise ValueError(f"alpha and beta must be in (0, 1], got alpha={alpha}, beta={beta}")
    level = values[0]
    trend = values[1] - values[0]
    result = [round(level, 6)]
    for v in values[1:]:
        prev_level = level
        level = alpha * v + (1.0 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1.0 - beta) * trend
        result.append(round(level, 6))
    return result


def exponential_moving_average(values: list[float], alpha: float = 0.3) -> list[float]:
    """Compute an exponential moving average (EMA) of *values*.

    Args:
        values: Time-ordered readings.
        alpha: Smoothing factor in (0, 1]. Higher values give more weight to recent data.

    Returns:
        EMA series of the same length as *values*.

    Raises:
        ValueError: If *alpha* is not in (0, 1].
    """
    if not values:
        return []
    if not (0 < alpha <= 1):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1.0 - alpha) * result[-1])
    return [round(x, 6) for x in result]


def daily_totals(values: list[float], period: int = 24) -> list[float]:
    """Aggregate *values* into totals over fixed *period*-length chunks.

    Args:
        values: Time-ordered readings (any length).
        period: Number of readings per chunk (default 24 for hourly data).

    Returns:
        List of chunk totals; the last partial chunk is included if it exists.
    """
    if not values:
        return []
    result: list[float] = []
    for i in range(0, len(values), period):
        result.append(round(sum(values[i : i + period]), 6))
    return result


def normalize_series(values: list[float]) -> list[float]:
    """Min-max normalise *values* to the [0, 1] range.

    Args:
        values: Numeric series.

    Returns:
        Normalised series of the same length. Returns all zeros for a constant series.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0] * len(values)
    return [round((v - lo) / (hi - lo), 6) for v in values]


def first_nonzero(values: list[float]) -> int:
    """Return the index of the first non-zero element in *values*.

    Args:
        values: Numeric series.

    Returns:
        0-based index of the first non-zero value, or -1 if all values are zero
        or the series is empty.
    """
    for i, v in enumerate(values):
        if v != 0.0:
            return i
    return -1


def cumulative_consumption(values: list[float]) -> list[float]:
    """Return the running cumulative sum of *values*.

    Alias for :func:`cumulative_sum` for energy-domain naming consistency.

    Args:
        values: Numeric series of energy readings.

    Returns:
        List of the same length where element i is sum(values[:i+1]).
    """
    return cumulative_sum(values)


def detect_trend_reversal(values: list[float], window: int = 5) -> list[int]:
    """Detect trend reversal points in a time series.

    A reversal is flagged at index *i* when the slope of the previous *window*
    values has opposite sign to the slope of the *window* values starting at *i*.

    Args:
        values: Time-ordered numeric readings.
        window: Look-back (and look-ahead) window size for slope estimation.

    Returns:
        Binary list of the same length; 1 at reversal points, 0 elsewhere.

    Raises:
        ValueError: If *values* is too short or *window* < 2.
    """
    if window < 2:
        raise ValueError(f"window must be at least 2, got {window}")
    n = len(values)
    if n < 2 * window:
        return [0] * n
    result = [0] * n
    arr = np.array(values, dtype=float)
    for i in range(window, n - window):
        before = arr[i - window : i]
        after = arr[i : i + window]
        slope_before = np.polyfit(np.arange(window), before, 1)[0]
        slope_after = np.polyfit(np.arange(window), after, 1)[0]
        if slope_before * slope_after < 0:
            result[i] = 1
    return result


def monthly_totals(daily_values: list[float], days_per_month: list[int] | None = None) -> list[float]:
    """Aggregate a daily series into monthly totals.

    Args:
        daily_values: Daily numeric readings.
        days_per_month: List of day-counts per month (default 12 months of 30 days).

    Returns:
        List of monthly sums.

    Raises:
        ValueError: If the total days in *days_per_month* exceeds len(*daily_values*).
    """
    if days_per_month is None:
        days_per_month = [30] * 12
    total = sum(days_per_month)
    if total > len(daily_values):
        raise ValueError(
            f"days_per_month total ({total}) exceeds available daily values ({len(daily_values)})"
        )
    result = []
    idx = 0
    for n_days in days_per_month:
        result.append(round(sum(daily_values[idx : idx + n_days]), 6))
        idx += n_days
    return result


def seasonal_variance(values: list[float], period: int = 24) -> float:
    """Compute the mean variance across seasonal buckets.

    Partitions *values* into *period* buckets (by index mod *period*) and returns
    the average of per-bucket variances, indicating how spread each seasonal
    slot is relative to its mean.

    Args:
        values: Time-ordered numeric readings.
        period: Seasonality period (e.g. 24 for hourly data).

    Returns:
        Mean per-bucket variance, rounded to 6 decimal places.

    Raises:
        ValueError: If *values* is empty or *period* < 1.
    """
    if not values:
        raise ValueError("values must not be empty")
    if period < 1:
        raise ValueError(f"period must be at least 1, got {period}")
    import statistics
    buckets: list[list[float]] = [[] for _ in range(period)]
    for i, v in enumerate(values):
        buckets[i % period].append(v)
    variances = [statistics.pvariance(b) for b in buckets if len(b) >= 2]
    if not variances:
        return 0.0
    return round(sum(variances) / len(variances), 6)


def trailing_average(values: list[float], n: int) -> list[float]:
    """Compute the trailing *n*-period average for each position.

    At position *i* the result is the mean of values[max(0, i-n+1):i+1].

    Args:
        values: Time-ordered numeric readings.
        n: Trailing window length (>= 1).

    Returns:
        Trailing averages list of the same length as *values*.

    Raises:
        ValueError: If *values* is empty or *n* < 1.
    """
    if not values:
        raise ValueError("values must not be empty")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    result = []
    for i in range(len(values)):
        chunk = values[max(0, i - n + 1) : i + 1]
        result.append(round(sum(chunk) / len(chunk), 6))
    return result


def detect_outlier_windows(
    values: list[float],
    window: int = 24,
    threshold_std: float = 2.0,
) -> list[tuple[int, int]]:
    """Detect windows with anomalously high mean consumption.

    Splits *values* into non-overlapping windows of size *window* and flags
    any window whose mean exceeds the global mean by more than *threshold_std*
    standard deviations.

    Args:
        values: Time-ordered numeric readings.
        window: Length of each non-overlapping assessment window.
        threshold_std: Number of global standard deviations above the global
            mean to use as the anomaly threshold.

    Returns:
        List of (start_index, end_index) tuples for flagged windows.

    Raises:
        ValueError: If *values* is empty or *window* < 1.
    """
    if not values:
        raise ValueError("values must not be empty")
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    arr = np.array(values, dtype=float)
    global_mean = float(arr.mean())
    global_std = float(arr.std()) if len(arr) > 1 else 0.0
    cutoff = global_mean + threshold_std * global_std
    flagged = []
    for start in range(0, len(values), window):
        chunk = values[start : start + window]
        if sum(chunk) / len(chunk) > cutoff:
            flagged.append((start, start + len(chunk) - 1))
    return flagged


def z_normalize(values: list[float]) -> list[float]:
    """Return a z-score normalised copy of *values*.

    Args:
        values: Input numeric sequence.

    Returns:
        List where each element is (v - mean) / std; returns zeros if std is 0.
    """
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5
    if std == 0:
        return [0.0] * len(values)
    return [round((v - mean) / std, 6) for v in values]


def mape(actual: list[float], predicted: list[float]) -> float:
    """Compute Mean Absolute Percentage Error (MAPE).

    Args:
        actual: Ground-truth values.
        predicted: Model predictions (same length as *actual*).

    Returns:
        MAPE as a percentage; pairs where actual==0 are skipped.
        Returns 0.0 if no valid pairs remain.
    """
    pairs = [
        abs(a - p) / abs(a) * 100
        for a, p in zip(actual, predicted, strict=False)
        if a != 0
    ]
    if not pairs:
        return 0.0
    return round(sum(pairs) / len(pairs), 4)


def hampel_filter(
    values: list[float], window: int = 5, n_sigma: float = 3.0
) -> list[float]:
    """Replace outliers detected by the Hampel identifier with the local median.

    Args:
        values: Time-ordered readings.
        window: Half-window size; the full window is 2*window+1.
        n_sigma: Number of scaled-MAD sigmas to consider an outlier.

    Returns:
        Cleaned series the same length as *values*.
    """
    if not values:
        return []
    k = 1.4826  # consistency constant for Gaussian
    result = list(values)
    for i in range(len(values)):
        lo = max(0, i - window)
        hi = min(len(values), i + window + 1)
        chunk = sorted(values[lo:hi])
        n = len(chunk)
        mid = n // 2
        median = chunk[mid] if n % 2 else (chunk[mid - 1] + chunk[mid]) / 2.0
        mad = k * sum(abs(v - median) for v in chunk) / n
        if mad > 0 and abs(values[i] - median) > n_sigma * mad:
            result[i] = median
    return [round(v, 6) for v in result]


def min_max_scale(
    values: list[float], feature_range: tuple[float, float] = (0.0, 1.0)
) -> list[float]:
    """Scale *values* to the given *feature_range* using min-max scaling.

    Args:
        values: Input numeric sequence.
        feature_range: Target (min, max) range; default (0, 1).

    Returns:
        Scaled list of the same length; all zeros if input is constant.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    span = hi - lo
    t_min, t_max = feature_range
    if span == 0:
        return [t_min] * len(values)
    return [
        round(t_min + (v - lo) / span * (t_max - t_min), 6) for v in values
    ]


def percent_change(values: list[float]) -> list[float]:
    """Compute the period-over-period percentage change for a time series.

    percent_change[i] = (values[i] - values[i-1]) / abs(values[i-1]) * 100

    Args:
        values: Time-ordered numeric observations (at least 2 values).

    Returns:
        List of length ``len(values) - 1`` with percentage changes.
        Returns empty list for inputs shorter than 2 elements.

    Raises:
        ValueError: If any denominator value is zero.
    """
    if len(values) < 2:
        return []
    result = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        if prev == 0.0:
            raise ValueError(f"Cannot compute percent change when previous value is zero (index {i - 1})")
        result.append(round((values[i] - prev) / abs(prev) * 100.0, 6))
    return result


def pair_difference(a: list[float], b: list[float]) -> list[float]:
    """Return element-wise difference (a[i] - b[i]) for two equal-length series.

    Args:
        a: First numeric series.
        b: Second numeric series (must have the same length as *a*).

    Returns:
        List of differences, rounded to 6 decimal places.

    Raises:
        ValueError: If series lengths differ.
    """
    if len(a) != len(b):
        raise ValueError(f"Series length mismatch: {len(a)} vs {len(b)}")
    return [round(x - y, 6) for x, y in zip(a, b)]
