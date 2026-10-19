"""Interval data aggregation utilities for energy time-series analysis."""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AggregatedInterval:
    """Statistics for an aggregated interval window."""

    count: int
    total: float
    mean: float
    minimum: float
    maximum: float
    std_dev: float
    median: float


def aggregate(values: list[float]) -> AggregatedInterval:
    """Compute summary statistics over a list of numeric interval readings.

    Args:
        values: Numeric readings (e.g. kWh per 15-min interval).

    Returns:
        :class:`AggregatedInterval` with count, total, mean, min, max,
        std_dev and median.

    Raises:
        ValueError: If values is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    return AggregatedInterval(
        count=len(values),
        total=round(sum(values), 6),
        mean=round(statistics.mean(values), 6),
        minimum=min(values),
        maximum=max(values),
        std_dev=round(statistics.pstdev(values), 6),
        median=round(statistics.median(values), 6),
    )


def resample_sum(values: list[float], bucket_size: int) -> list[float]:
    """Down-sample a high-resolution series by summing into larger buckets.

    For example, resample 96 × 15-min readings into 24 × hourly totals by
    passing ``bucket_size=4``.

    Args:
        values: Original high-resolution readings.
        bucket_size: Number of readings to sum into each output bucket.

    Returns:
        List of summed buckets.  The final bucket is included even when
        ``len(values)`` is not a multiple of ``bucket_size``.

    Raises:
        ValueError: If bucket_size < 1 or values is empty.
    """
    if bucket_size < 1:
        raise ValueError("bucket_size must be >= 1")
    if not values:
        raise ValueError("values must not be empty")
    result: list[float] = []
    for i in range(0, len(values), bucket_size):
        result.append(round(sum(values[i : i + bucket_size]), 6))
    return result


def resample_mean(values: list[float], bucket_size: int) -> list[float]:
    """Down-sample a high-resolution series by averaging into larger buckets.

    Args:
        values: Original high-resolution readings.
        bucket_size: Number of readings per output bucket.

    Returns:
        List of averaged bucket values.

    Raises:
        ValueError: If bucket_size < 1 or values is empty.
    """
    if bucket_size < 1:
        raise ValueError("bucket_size must be >= 1")
    if not values:
        raise ValueError("values must not be empty")
    result: list[float] = []
    for i in range(0, len(values), bucket_size):
        chunk = values[i : i + bucket_size]
        result.append(round(sum(chunk) / len(chunk), 6))
    return result


def rolling_average(values: list[float], window: int) -> list[float]:
    """Compute a simple trailing rolling average over an interval series.

    Args:
        values: Input readings.
        window: Number of periods in the rolling window (>= 1).

    Returns:
        List of rolling averages, same length as input.  Values before the
        window fills up use the available readings (expanding window).

    Raises:
        ValueError: If window < 1 or values is empty.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if not values:
        raise ValueError("values must not be empty")
    result: list[float] = []
    for i, _ in enumerate(values):
        chunk = values[max(0, i - window + 1) : i + 1]
        result.append(round(sum(chunk) / len(chunk), 6))
    return result


def percentile(values: list[float], p: float) -> float:
    """Return the p-th percentile of a list of values using linear interpolation.

    Args:
        values: Numeric values (unsorted is fine).
        p: Percentile in [0, 100].

    Returns:
        Interpolated percentile value.

    Raises:
        ValueError: If values is empty or p is outside [0, 100].
    """
    if not values:
        raise ValueError("values must not be empty")
    if not (0.0 <= p <= 100.0):
        raise ValueError("p must be in [0, 100]")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    rank = p / 100.0 * (n - 1)
    lower = int(rank)
    upper = lower + 1
    if upper >= n:
        return sorted_vals[-1]
    frac = rank - lower
    return round(sorted_vals[lower] + frac * (sorted_vals[upper] - sorted_vals[lower]), 6)


def detect_outliers_iqr(
    values: list[float],
    multiplier: float = 1.5,
) -> list[int]:
    """Identify outlier indices using the IQR fence method.

    Args:
        values: Numeric readings.
        multiplier: IQR multiplier for the fence (default 1.5 = Tukey fence).

    Returns:
        List of indices where the reading is outside the IQR fences.

    Raises:
        ValueError: If values has fewer than 4 elements.
    """
    if len(values) < 4:
        raise ValueError("Need at least 4 values to detect outliers via IQR")
    q1 = percentile(values, 25)
    q3 = percentile(values, 75)
    iqr = q3 - q1
    low_fence = q1 - multiplier * iqr
    high_fence = q3 + multiplier * iqr
    return [i for i, v in enumerate(values) if v < low_fence or v > high_fence]
