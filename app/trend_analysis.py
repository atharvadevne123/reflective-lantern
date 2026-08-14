"""Trend analysis utilities for energy time-series data."""

from __future__ import annotations

import logging
import math
from typing import NamedTuple

logger = logging.getLogger(__name__)


class TrendResult(NamedTuple):
    slope: float
    intercept: float
    direction: str  # "rising" | "falling" | "stable"
    r_squared: float


def linear_trend(values: list[float]) -> TrendResult:
    """Fit a least-squares line to *values* (index as x-axis).

    Returns slope, intercept, direction label, and R².
    """
    n = len(values)
    if n < 2:
        return TrendResult(slope=0.0, intercept=values[0] if values else 0.0, direction="stable", r_squared=0.0)
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values, strict=False))
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    slope = ss_xy / ss_xx if ss_xx != 0 else 0.0
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in values)
    if ss_tot == 0:
        r_sq = 1.0
    else:
        residuals = [y - (slope * x + intercept) for x, y in zip(xs, values, strict=False)]
        ss_res = sum(r**2 for r in residuals)
        r_sq = 1 - ss_res / ss_tot
    if slope > 0.01:
        direction = "rising"
    elif slope < -0.01:
        direction = "falling"
    else:
        direction = "stable"
    return TrendResult(
        slope=round(slope, 6), intercept=round(intercept, 6), direction=direction, r_squared=round(r_sq, 6)
    )


def percentage_change(old: float, new: float) -> float:
    """Return percentage change from *old* to *new*. Returns 0.0 if old is zero."""
    if old == 0:
        return 0.0
    return round((new - old) / abs(old) * 100, 4)


def rolling_mean(values: list[float], window: int) -> list[float]:
    """Compute a simple rolling mean with the given *window* size.

    Pads missing values at the start with the first available rolling mean.
    """
    if not values or window <= 0:
        return []
    result: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        result.append(sum(chunk) / len(chunk))
    return [round(v, 6) for v in result]


def detect_change_points(values: list[float], threshold: float = 2.0) -> list[int]:
    """Return indices where the difference exceeds *threshold* standard deviations.

    Uses a simple z-score on first-differences to flag abrupt shifts.
    """
    if len(values) < 3:
        return []
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    mean_d = sum(diffs) / len(diffs)
    variance = sum((d - mean_d) ** 2 for d in diffs) / len(diffs)
    std_d = math.sqrt(variance) if variance > 0 else 0.0
    if std_d == 0:
        return []
    cps = [i + 1 for i, d in enumerate(diffs) if abs((d - mean_d) / std_d) > threshold]
    if cps:
        logger.debug("detect_change_points: found %d change point(s) in series of length %d", len(cps), len(values))
    return cps


def seasonal_decompose_naive(values: list[float], period: int) -> dict[str, list[float]]:
    """Naive additive seasonal decomposition (trend + seasonal + residual).

    Uses a rolling mean of length *period* as the trend component.
    """
    if len(values) < period * 2:
        return {"trend": list(values), "seasonal": [0.0] * len(values), "residual": [0.0] * len(values)}
    trend = rolling_mean(values, period)
    detrended = [v - t for v, t in zip(values, trend, strict=False)]
    seasonal: list[float] = [0.0] * len(values)
    for s in range(period):
        idxs = list(range(s, len(values), period))
        avg = sum(detrended[i] for i in idxs) / len(idxs)
        for i in idxs:
            seasonal[i] = avg
    residual = [v - t - s for v, t, s in zip(values, trend, seasonal, strict=False)]
    return {
        "trend": [round(x, 4) for x in trend],
        "seasonal": [round(x, 4) for x in seasonal],
        "residual": [round(x, 4) for x in residual],
    }


def year_over_year_growth(monthly_series: list[float], period: int = 12) -> list[float]:
    """Compute year-over-year growth rates for a monthly consumption series.

    Args:
        monthly_series: List of monthly values (at least 2*period elements).
        period: Number of periods in one year (default 12 for monthly data).

    Returns:
        List of YoY growth percentages starting from index *period*.
        Returns an empty list if the series is too short.
    """
    if len(monthly_series) < period + 1:
        return []
    result: list[float] = []
    for i in range(period, len(monthly_series)):
        prev = monthly_series[i - period]
        curr = monthly_series[i]
        result.append(percentage_change(prev, curr))
    return result


def rate_of_change(values: list[float], lag: int = 1) -> list[float]:
    """Compute the rate of change between values separated by *lag* periods.

    Rate of change = (value[i] - value[i-lag]) / abs(value[i-lag]) * 100
    when value[i-lag] != 0, else 0.0.

    Args:
        values: Time-ordered readings.
        lag: Number of periods to look back (default 1).

    Returns:
        List of percentage changes, length = len(values) - lag.
        Empty list if len(values) <= lag.
    """
    if lag < 1 or len(values) <= lag:
        return []
    return [percentage_change(values[i - lag], values[i]) for i in range(lag, len(values))]


def momentum_score(
    values: list[float],
    short_window: int = 7,
    long_window: int = 30,
) -> dict[str, float]:
    """Compute a momentum score by comparing short and long moving averages.

    A positive score indicates short-term consumption is rising faster than the
    long-term baseline (bullish momentum); negative is the reverse.

    Args:
        values: Time-ordered consumption readings (kWh).
        short_window: Periods for the short-term moving average (default 7).
        long_window: Periods for the long-term moving average (default 30).

    Returns:
        Dict with 'short_ma', 'long_ma', 'momentum', and 'signal'
        ('increasing'|'decreasing'|'neutral').

    Raises:
        ValueError: If *values* is shorter than *long_window* or windows are invalid.
    """
    if short_window < 1 or long_window < 1:
        raise ValueError("Window sizes must be >= 1")
    if short_window >= long_window:
        raise ValueError(f"short_window ({short_window}) must be less than long_window ({long_window})")
    if len(values) < long_window:
        raise ValueError(f"Need at least {long_window} values, got {len(values)}")
    short_ma = sum(values[-short_window:]) / short_window
    long_ma = sum(values[-long_window:]) / long_window
    momentum = round(short_ma - long_ma, 4)
    relative_threshold = long_ma * 0.01 if long_ma != 0 else 0.0
    if momentum > relative_threshold:
        signal = "increasing"
    elif momentum < -relative_threshold:
        signal = "decreasing"
    else:
        signal = "neutral"
    return {
        "short_ma": round(short_ma, 4),
        "long_ma": round(long_ma, 4),
        "momentum": momentum,
        "signal": signal,
    }


def cumulative_sum(values: list[float]) -> list[float]:
    """Return the cumulative sum of *values* (CUSUM).

    Useful for detecting long-run drift and structural shifts in a series.

    Args:
        values: Time-ordered readings.

    Returns:
        Cumulative sum list of the same length as *values*.
    """
    result: list[float] = []
    total = 0.0
    for v in values:
        total += v
        result.append(round(total, 6))
    return result


__all__ = [
    "TrendResult",
    "autocorrelation",
    "cumulative_sum",
    "detect_change_points",
    "detect_outliers",
    "double_exponential_smoothing",
    "exponential_growth_rate",
    "linear_regression_trend",
    "linear_trend",
    "momentum_score",
    "percentage_change",
    "period_comparison",
    "rate_of_change",
    "rolling_mean",
    "seasonal_decompose_naive",
    "smoothed_trend",
    "trend_reversal_count",
    "trend_strength",
    "trend_summary",
    "year_over_year_growth",
]


def exponential_weighted_mean(values: list[float], alpha: float = 0.3) -> list[float]:
    """Compute exponential weighted moving average of a series.

    Args:
        values: Input numeric series.
        alpha: Smoothing factor in (0, 1].

    Returns:
        List of EWMA values, same length as input.

    Raises:
        ValueError: If alpha is not in (0, 1].
    """
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")
    if not values:
        return []
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return [round(x, 6) for x in result]


def trend_strength(values: list[float]) -> float:
    """Estimate trend strength as R-squared of a linear fit.

    Args:
        values: Numeric series with at least 2 elements.

    Returns:
        R-squared value in [0, 1] where 1 = perfect linear trend.

    Raises:
        ValueError: If fewer than 2 values are provided.
    """
    if len(values) < 2:
        raise ValueError("Need at least 2 values")
    n = len(values)
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    ss_tot = sum((v - y_mean) ** 2 for v in values)
    if ss_tot == 0:
        return 1.0
    x_dev = [i - x_mean for i in range(n)]
    y_dev = [v - y_mean for v in values]
    ss_xx = sum(d * d for d in x_dev)
    slope = sum(xd * yd for xd, yd in zip(x_dev, y_dev, strict=False)) / ss_xx if ss_xx else 0.0
    ss_res = sum((values[i] - (y_mean + slope * x_dev[i])) ** 2 for i in range(n))
    return round(max(0.0, 1.0 - ss_res / ss_tot), 6)


def peak_valley_count(values: list[float]) -> dict[str, int]:
    """Count local peaks and valleys in a series.

    Args:
        values: Numeric series.

    Returns:
        Dict with 'peaks' and 'valleys' counts.
    """
    if len(values) < 3:
        return {"peaks": 0, "valleys": 0}
    peaks = valleys = 0
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            peaks += 1
        elif values[i] < values[i - 1] and values[i] < values[i + 1]:
            valleys += 1
    return {"peaks": peaks, "valleys": valleys}


def normalised_range(values: list[float]) -> float:
    """Return (max - min) / mean of a series, a relative spread metric.

    Args:
        values: Numeric series with at least 1 element.

    Returns:
        Normalised range as a float.

    Raises:
        ValueError: If values is empty or mean is zero.
    """
    if not values:
        raise ValueError("values must be non-empty")
    mean = sum(values) / len(values)
    if mean == 0:
        raise ValueError("mean is zero, cannot normalise")
    return round((max(values) - min(values)) / mean, 6)


def autocorrelation(values: list[float], lag: int = 1) -> float:
    """Compute the Pearson autocorrelation at a given *lag*.

    Args:
        values: Time-ordered numeric values (at least lag+2 elements).
        lag: Number of periods to shift.

    Returns:
        Autocorrelation coefficient in [-1, 1]; 0.0 if insufficient data.
    """
    n = len(values)
    if n <= lag + 1:
        return 0.0
    mean = sum(values) / n
    denom = sum((v - mean) ** 2 for v in values)
    if denom == 0:
        return 0.0
    numer = sum((values[i] - mean) * (values[i - lag] - mean) for i in range(lag, n))
    return round(numer / denom, 6)


def double_exponential_smoothing(values: list[float], alpha: float = 0.3, beta: float = 0.1) -> list[float]:
    """Apply Holt's double exponential smoothing (level + trend).

    Args:
        values: Time-ordered numeric values (at least 2 elements).
        alpha: Smoothing factor for the level (0 < alpha < 1).
        beta: Smoothing factor for the trend (0 < beta < 1).

    Returns:
        Smoothed series of the same length as *values*.
    """
    if len(values) < 2:
        return list(values)
    level = values[0]
    trend = values[1] - values[0]
    result = [round(level + trend, 6)]
    for v in values[1:]:
        prev_level = level
        level = alpha * v + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
        result.append(round(level + trend, 6))
    return result


def trend_reversal_count(values: list[float]) -> int:
    """Count the number of direction changes (reversals) in a time series.

    A reversal occurs when a value increases after a decrease or vice versa.
    Single-element or empty series have zero reversals.

    Args:
        values: Time-ordered numeric observations.

    Returns:
        Non-negative integer count of direction changes.
    """
    if len(values) < 3:
        return 0
    count = 0
    for i in range(1, len(values) - 1):
        prev_diff = values[i] - values[i - 1]
        next_diff = values[i + 1] - values[i]
        if prev_diff * next_diff < 0:
            count += 1
    return count


def exponential_growth_rate(values: list[float]) -> float:
    """Estimate the average continuous exponential growth rate of a series.

    Computed as (ln(last) - ln(first)) / (n - 1), which approximates the
    per-period log growth rate.

    Args:
        values: Non-empty list of strictly positive values.

    Returns:
        Estimated per-period exponential growth rate, rounded to 6 decimal places.

    Raises:
        ValueError: If *values* is empty, has fewer than 2 elements, or contains
            non-positive values.
    """
    import math

    if len(values) < 2:
        raise ValueError("values must have at least 2 elements")
    if any(v <= 0 for v in values):
        raise ValueError("All values must be strictly positive for exponential growth rate")
    return round((math.log(values[-1]) - math.log(values[0])) / (len(values) - 1), 6)


def period_comparison(
    series_a: list[float],
    series_b: list[float],
) -> dict[str, float]:
    """Compare two equal-length time periods and return summary statistics.

    Computes the mean, total, and percentage change between two periods.
    Useful for month-over-month or year-over-year comparisons.

    Args:
        series_a: Reference period values (e.g. previous month).
        series_b: Comparison period values (e.g. current month).

    Returns:
        Dict with keys: mean_a, mean_b, total_a, total_b, pct_change.
        pct_change is positive when series_b > series_a.

    Raises:
        ValueError: If either series is empty or they have different lengths.
    """
    if not series_a or not series_b:
        raise ValueError("Both series must be non-empty")
    if len(series_a) != len(series_b):
        raise ValueError(f"Series lengths must match: {len(series_a)} vs {len(series_b)}")
    mean_a = sum(series_a) / len(series_a)
    mean_b = sum(series_b) / len(series_b)
    total_a = sum(series_a)
    total_b = sum(series_b)
    pct_change = round(100.0 * (mean_b - mean_a) / mean_a, 4) if mean_a != 0 else 0.0
    return {
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "total_a": round(total_a, 4),
        "total_b": round(total_b, 4),
        "pct_change": pct_change,
    }


def smoothed_trend(
    values: list[float],
    window: int = 7,
) -> dict[str, list[float]]:
    """Apply rolling mean smoothing and return both the smoothed series and residuals.

    Residuals are the difference between raw values and their smoothed counterparts,
    useful for isolating noise from the underlying trend.

    Args:
        values: Input time series.
        window: Rolling window size (must be >= 2 and <= len(values)).

    Returns:
        Dict with 'smoothed' (rolling mean series) and 'residuals' (values - smoothed).
        Both have the same length as *values*.

    Raises:
        ValueError: If *values* has fewer than 2 elements or *window* < 2.
    """
    if len(values) < 2:
        raise ValueError("values must have at least 2 elements")
    if window < 2:
        raise ValueError(f"window must be at least 2, got {window}")
    smoothed = rolling_mean(values, window)
    residuals = [round(v - s, 6) for v, s in zip(values, smoothed, strict=False)]
    return {"smoothed": smoothed, "residuals": residuals}


def trend_summary(values: list[float]) -> dict[str, object]:
    """Compute a concise trend summary combining direction, strength, and change points.

    Args:
        values: Input time series (at least 4 elements recommended).

    Returns:
        Dict with 'direction' ('up'|'down'|'flat'), 'strength' (float 0-1),
        'change_points' (list[int]), and 'pct_change_overall' (float).

    Raises:
        ValueError: If *values* has fewer than 2 elements.
    """
    if len(values) < 2:
        raise ValueError("values must have at least 2 elements")
    slope = linear_trend(values).slope
    direction = "flat" if abs(slope) < 1e-9 else ("up" if slope > 0 else "down")
    strength = trend_strength(values) if len(values) >= 4 else 0.0
    change_pts = detect_change_points(values) if len(values) >= 4 else []
    first, last = values[0], values[-1]
    if first == 0.0:
        pct_change = 0.0
    else:
        pct_change = round((last - first) / abs(first) * 100.0, 4)
    return {
        "direction": direction,
        "strength": round(strength, 4),
        "change_points": change_pts,
        "pct_change_overall": pct_change,
    }


def detect_outliers(values: list[float], z_threshold: float = 2.5) -> list[int]:
    """Return indices of values more than *z_threshold* standard deviations from the mean.

    Args:
        values: Input time series.
        z_threshold: Number of standard deviations to use as the cutoff.

    Returns:
        Sorted list of 0-based indices where the z-score exceeds *z_threshold*.
        Returns an empty list for series with fewer than 2 elements or zero variance.
    """
    n = len(values)
    if n < 2:
        return []
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    if variance == 0.0:
        return []
    std = variance**0.5
    return [i for i, v in enumerate(values) if abs(v - mean) / std > z_threshold]


def linear_regression_trend(values: list[float]) -> dict[str, float]:
    """Fit a simple linear regression to *values* and return coefficients.

    The independent variable is the integer index (0, 1, 2, …).

    Args:
        values: Time series of at least 2 observations.

    Returns:
        Dict with 'slope', 'intercept', and 'r_squared'.
        Returns zeros for series with fewer than 2 points.
    """
    n = len(values)
    if n < 2:
        return {"slope": 0.0, "intercept": float(values[0]) if values else 0.0, "r_squared": 0.0}
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    ss_xy = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    ss_xx = sum((i - x_mean) ** 2 for i in range(n))
    slope = ss_xy / ss_xx if ss_xx != 0 else 0.0
    intercept = y_mean - slope * x_mean
    y_pred = [slope * i + intercept for i in range(n)]
    ss_tot = sum((v - y_mean) ** 2 for v in values)
    ss_res = sum((v - p) ** 2 for v, p in zip(values, y_pred, strict=False))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
    return {"slope": round(slope, 6), "intercept": round(intercept, 6), "r_squared": round(r_squared, 6)}
