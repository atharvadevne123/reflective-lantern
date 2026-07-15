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
