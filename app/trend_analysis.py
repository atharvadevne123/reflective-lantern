"""Trend analysis utilities for energy time-series data."""

from __future__ import annotations

import math
from typing import NamedTuple


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
        ss_res = sum(r ** 2 for r in residuals)
        r_sq = 1 - ss_res / ss_tot
    if slope > 0.01:
        direction = "rising"
    elif slope < -0.01:
        direction = "falling"
    else:
        direction = "stable"
    return TrendResult(slope=round(slope, 6), intercept=round(intercept, 6), direction=direction, r_squared=round(r_sq, 6))


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
    return [i + 1 for i, d in enumerate(diffs) if abs((d - mean_d) / std_d) > threshold]


def seasonal_decompose_naive(
    values: list[float], period: int
) -> dict[str, list[float]]:
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

__all__ = [
    "TrendResult",
    "detect_change_points",
    "linear_trend",
    "percentage_change",
    "rolling_mean",
    "seasonal_decompose_naive",
]
