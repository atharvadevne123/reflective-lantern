"""Energy efficiency reporting: baseline comparison and savings estimation."""

from __future__ import annotations

import logging

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


def peak_demand_report(hourly_kwh: list[float]) -> dict[str, object]:
    """Identify peak demand windows from hourly consumption.

    Args:
        hourly_kwh: 24 (or more) hourly consumption values.

    Returns:
        Dict with peak_hour, peak_kwh, off_peak_mean, and demand_factor.
    """
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
