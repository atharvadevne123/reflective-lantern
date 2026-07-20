"""Energy efficiency reporting: baseline comparison and savings estimation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def estimate_savings(
    actual_kwh: list[float],
    baseline_kwh: list[float],
    tariff_per_kwh: float = 0.15,
) -> dict[str, float]:
    """Estimate energy savings vs a baseline consumption series.

    Args:
        actual_kwh: Measured consumption values.
        baseline_kwh: Reference (baseline) consumption values of same length.
        tariff_per_kwh: Electricity cost in currency per kWh.

    Returns:
        Dict with total_saved_kwh, total_saved_cost, and savings_pct.
    """
    if len(actual_kwh) != len(baseline_kwh):
        raise ValueError(
            f"actual_kwh and baseline_kwh must be the same length (got {len(actual_kwh)} vs {len(baseline_kwh)})"
        )
    actual = np.array(actual_kwh, dtype=float)
    baseline = np.array(baseline_kwh, dtype=float)
    saved = baseline - actual
    total_saved_kwh = float(saved.sum())
    total_saved_cost = round(total_saved_kwh * tariff_per_kwh, 2)
    baseline_total = float(baseline.sum())
    savings_pct = round(100.0 * total_saved_kwh / baseline_total, 2) if baseline_total > 0 else 0.0
    logger.info("Savings: %.2f kWh  $%.2f  %.1f%%", total_saved_kwh, total_saved_cost, savings_pct)
    return {
        "total_saved_kwh": round(total_saved_kwh, 3),
        "total_saved_cost": total_saved_cost,
        "savings_pct": savings_pct,
    }


def peak_demand_report(hourly_kwh: list[float]) -> dict[str, Any]:
    """Identify peak demand windows from hourly consumption.

    Args:
        hourly_kwh: 24 (or more) hourly consumption values.

    Returns:
        Dict with peak_hour, peak_kwh, off_peak_mean, and demand_factor.

    Raises:
        ValueError: If *hourly_kwh* is empty.
    """
    if not hourly_kwh:
        raise ValueError("hourly_kwh must not be empty")
    arr = np.array(hourly_kwh, dtype=float)
    peak_idx = int(np.argmax(arr))
    peak_kwh = float(arr[peak_idx])
    mean_kwh = float(arr.mean())
    demand_factor = round(peak_kwh / mean_kwh, 3) if mean_kwh > 0 else 0.0
    return {
        "peak_hour": peak_idx,
        "peak_kwh": round(peak_kwh, 3),
        "off_peak_mean": round(mean_kwh, 3),
        "demand_factor": demand_factor,
    }


_GRADE_THRESHOLDS: list[tuple[float, str]] = [
    (0.20, "A+"),
    (0.10, "A"),
    (0.05, "A-"),
    (0.0, "B"),
    (-0.05, "C"),
    (-0.15, "D"),
]


def energy_efficiency_grade(actual_kwh: float, baseline_kwh: float) -> str:
    """Return a letter grade (A+…F) based on consumption vs baseline.

    Args:
        actual_kwh: Measured consumption.
        baseline_kwh: Reference baseline consumption (must be positive).

    Returns:
        Grade string in {'A+', 'A', 'A-', 'B', 'C', 'D', 'F'}.
    """
    if baseline_kwh <= 0:
        return "F"
    reduction = (baseline_kwh - actual_kwh) / baseline_kwh
    for threshold, grade in _GRADE_THRESHOLDS:
        if reduction >= threshold:
            return grade
    return "F"


def monthly_consumption_summary(
    daily_kwh: list[float],
    tariff_per_kwh: float = 0.15,
) -> dict[str, Any]:
    """Summarise a month of daily consumption readings.

    Args:
        daily_kwh: List of daily consumption values (kWh); typically 28-31 entries.
        tariff_per_kwh: Electricity cost per kWh for cost estimation.

    Returns:
        Dict with total_kwh, mean_kwh, max_kwh, min_kwh, std_kwh, estimated_cost.

    Raises:
        ValueError: If *daily_kwh* is empty.
    """
    if not daily_kwh:
        raise ValueError("daily_kwh must not be empty")
    arr = np.array(daily_kwh, dtype=float)
    total = float(arr.sum())
    return {
        "total_kwh": round(total, 3),
        "mean_kwh": round(float(arr.mean()), 3),
        "max_kwh": round(float(arr.max()), 3),
        "min_kwh": round(float(arr.min()), 3),
        "std_kwh": round(float(arr.std()), 3),
        "estimated_cost": round(total * tariff_per_kwh, 2),
        "days": len(daily_kwh),
    }


def seasonal_efficiency_score(
    actual_kwh: list[float],
    baseline_kwh: list[float],
    season_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute a weighted efficiency score across seasonal segments.

    Divides the series into four equal seasonal quarters and computes
    savings pct per quarter, then applies optional weights.

    Args:
        actual_kwh: Measured consumption series (must match baseline length).
        baseline_kwh: Baseline consumption series.
        season_weights: Optional mapping of quarter name to weight. Defaults to
            equal weights for 'q1', 'q2', 'q3', 'q4'.

    Returns:
        Dict with per-quarter savings pct and a weighted 'overall_score'.

    Raises:
        ValueError: If series lengths don't match or are empty.
    """
    if len(actual_kwh) != len(baseline_kwh):
        raise ValueError("actual_kwh and baseline_kwh must have the same length")
    if not actual_kwh:
        raise ValueError("Input series must not be empty")
    weights = season_weights or {"q1": 0.25, "q2": 0.25, "q3": 0.25, "q4": 0.25}
    n = len(actual_kwh)
    quarter = max(1, n // 4)
    quarters = {
        "q1": (0, quarter),
        "q2": (quarter, 2 * quarter),
        "q3": (2 * quarter, 3 * quarter),
        "q4": (3 * quarter, n),
    }
    result: dict[str, float] = {}
    weighted_sum = 0.0
    for name, (start, end) in quarters.items():
        a = np.array(actual_kwh[start:end], dtype=float)
        b = np.array(baseline_kwh[start:end], dtype=float)
        b_total = float(b.sum())
        pct = round(100.0 * float((b - a).sum()) / b_total, 2) if b_total > 0 else 0.0
        result[f"{name}_savings_pct"] = pct
        weighted_sum += pct * weights.get(name, 0.25)
    result["overall_score"] = round(weighted_sum, 2)
    return result


def consumption_trend(daily_kwh: list[float]) -> str:
    """Classify consumption trend over a period as 'rising', 'falling', or 'stable'.

    Fits a simple linear regression to *daily_kwh* and classifies the slope.

    Args:
        daily_kwh: List of daily consumption values (kWh).

    Returns:
        One of 'rising', 'falling', or 'stable'.
    """
    if len(daily_kwh) < 2:
        return "stable"
    arr = np.array(daily_kwh, dtype=float)
    x = np.arange(len(arr), dtype=float)
    x_mean, y_mean = x.mean(), arr.mean()
    denom = float(((x - x_mean) ** 2).sum())
    if denom < 1e-9:
        return "stable"
    slope = float(((x - x_mean) * (arr - y_mean)).sum() / denom)
    relative = abs(slope) / (y_mean + 1e-9)
    if relative < 0.01:
        return "stable"
    return "rising" if slope > 0 else "falling"


def peak_demand_by_period(
    hourly_kwh: list[float],
    period_hours: int = 4,
) -> list[dict[str, object]]:
    """Aggregate hourly consumption into fixed-length periods and find peak.

    Args:
        hourly_kwh: List of hourly consumption values.
        period_hours: Length of each period in hours (default 4, i.e. 6 periods/day).

    Returns:
        List of dicts with 'period_start', 'total_kwh', and 'is_peak' keys.
        'is_peak' is True for the period with the highest total consumption.

    Raises:
        ValueError: If *hourly_kwh* is empty or *period_hours* < 1.
    """
    if not hourly_kwh:
        raise ValueError("hourly_kwh must not be empty")
    if period_hours < 1:
        raise ValueError(f"period_hours must be at least 1, got {period_hours}")
    periods = []
    for start in range(0, len(hourly_kwh), period_hours):
        chunk = hourly_kwh[start : start + period_hours]
        periods.append({"period_start": start, "total_kwh": round(sum(chunk), 3), "is_peak": False})
    if periods:
        max_total = max(p["total_kwh"] for p in periods)
        for p in periods:
            if p["total_kwh"] == max_total:
                p["is_peak"] = True
                break
    return periods


def consumption_efficiency_ratio(
    actual_kwh: float,
    target_kwh: float,
) -> float:
    """Return the ratio of actual to target energy consumption.

    A value < 1.0 means the building consumed less than the target (efficient).
    A value > 1.0 means consumption exceeded the target (inefficient).

    Args:
        actual_kwh: Measured energy consumption in kWh.
        target_kwh: Target or baseline consumption in kWh.

    Returns:
        Ratio actual/target, or 0.0 if target is non-positive.
    """
    if target_kwh <= 0:
        return 0.0
    return round(actual_kwh / target_kwh, 4)


def daily_average_consumption(hourly_series: list[float]) -> float:
    """Compute the average daily energy consumption from an hourly time series.

    Args:
        hourly_series: Hourly kWh readings (any length).

    Returns:
        Mean daily consumption (sum over 24-hour blocks) or 0.0 for empty input.
    """
    if not hourly_series:
        return 0.0
    days = max(len(hourly_series) / 24.0, 1.0)
    return round(sum(hourly_series) / days, 4)

__all__ = [
    "estimate_savings",
    "peak_demand_report",
    "energy_efficiency_grade",
    "monthly_consumption_summary",
    "seasonal_efficiency_score",
    "consumption_trend",
    "peak_demand_by_period",
    "consumption_efficiency_ratio",
    "daily_average_consumption",
]
