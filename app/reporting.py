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


def benchmark_vs_portfolio(
    building_kwh: float,
    portfolio_kwh: list[float],
) -> dict[str, object]:
    """Compare a single building's consumption against a portfolio of peers.

    Args:
        building_kwh: Target building's daily energy consumption in kWh.
        portfolio_kwh: Peer buildings' daily consumption values.

    Returns:
        Dict with 'percentile' (0-100), 'rank' (1-indexed), 'total_peers',
        'is_above_median', and 'grade' (A if bottom 25%, D if top 25%).

    Raises:
        ValueError: If *portfolio_kwh* is empty.
    """
    if not portfolio_kwh:
        raise ValueError("portfolio_kwh must not be empty")
    arr = np.array(portfolio_kwh, dtype=float)
    n = len(arr)
    rank = int((arr <= building_kwh).sum())
    percentile = round(rank / n * 100.0, 2)
    median = float(np.median(arr))
    if percentile <= 25:
        grade = "A"
    elif percentile <= 50:
        grade = "B"
    elif percentile <= 75:
        grade = "C"
    else:
        grade = "D"
    return {
        "percentile": percentile,
        "rank": rank,
        "total_peers": n,
        "is_above_median": bool(building_kwh > median),
        "grade": grade,
    }


__all__ = [
    "benchmark_vs_portfolio",
    "consumption_efficiency_ratio",
    "consumption_trend",
    "daily_average_consumption",
    "energy_efficiency_grade",
    "estimate_savings",
    "hourly_cost_breakdown",
    "kwh_to_wh",
    "monthly_consumption_summary",
    "peak_demand_by_period",
    "peak_demand_report",
    "rolling_savings_summary",
    "savings_report",
    "seasonal_efficiency_score",
    "tariff_cost",
    "top_consumption_hours",
    "wh_to_kwh",
]


def kwh_to_wh(kwh: float) -> float:
    """Convert kilowatt-hours to watt-hours.

    Args:
        kwh: Energy in kilowatt-hours.

    Returns:
        Equivalent energy in watt-hours.
    """
    return round(kwh * 1000.0, 4)


def wh_to_kwh(wh: float) -> float:
    """Convert watt-hours to kilowatt-hours.

    Args:
        wh: Energy in watt-hours.

    Returns:
        Equivalent energy in kilowatt-hours.
    """
    return round(wh / 1000.0, 6)


def tariff_cost(kwh: float, tariff_per_kwh: float) -> float:
    """Compute energy cost for a given consumption and tariff rate.

    Args:
        kwh: Energy consumption in kilowatt-hours.
        tariff_per_kwh: Electricity tariff in currency per kWh.

    Returns:
        Total cost rounded to 4 decimal places.
        Returns 0.0 for non-positive kwh.
    """
    if kwh <= 0:
        return 0.0
    return round(kwh * tariff_per_kwh, 4)


def summarize_energy_period(readings: list[float], label: str = "period") -> dict:
    """Return a summary dict for an energy readings period.

    Args:
        readings: List of energy readings in kWh.
        label: Human-readable period label.

    Returns:
        Dict with label, count, total, mean, min, max.

    Raises:
        ValueError: If readings is empty.
    """
    if not readings:
        raise ValueError("readings must be non-empty")
    total = sum(readings)
    return {
        "label": label,
        "count": len(readings),
        "total": round(total, 4),
        "mean": round(total / len(readings), 4),
        "min": min(readings),
        "max": max(readings),
    }


def format_report_row(data: dict, fields: list[str], separator: str = ",") -> str:
    """Format a data dict as a delimited row string.

    Args:
        data: Dict of field names to values.
        fields: Ordered list of field names to include.
        separator: Delimiter character.

    Returns:
        Delimited string of field values.
    """
    return separator.join(str(data.get(f, "")) for f in fields)


def aggregate_daily_report(hourly: list[float]) -> list[float]:
    """Aggregate hourly readings into daily totals.

    Args:
        hourly: List of hourly energy readings (kWh). Length need not be multiple of 24.

    Returns:
        List of daily totals (last partial day included if any).
    """
    if not hourly:
        return []
    days = []
    for i in range(0, len(hourly), 24):
        days.append(round(sum(hourly[i : i + 24]), 4))
    return days


def report_anomaly_summary(flags: list[bool]) -> dict:
    """Summarise a boolean anomaly flag sequence.

    Args:
        flags: List of booleans where True = anomaly.

    Returns:
        Dict with total, anomaly_count, normal_count, anomaly_rate.

    Raises:
        ValueError: If flags is empty.
    """
    if not flags:
        raise ValueError("flags must be non-empty")
    anomaly_count = sum(flags)
    return {
        "total": len(flags),
        "anomaly_count": anomaly_count,
        "normal_count": len(flags) - anomaly_count,
        "anomaly_rate": round(anomaly_count / len(flags), 4),
    }


def top_consumption_hours(
    hourly_kwh: list[float],
    n: int = 5,
) -> list[dict[str, Any]]:
    """Return the *n* hours with the highest consumption in *hourly_kwh*.

    Args:
        hourly_kwh: List of hourly energy readings (index 0 = hour 0).
        n: Number of top hours to return (capped at len(hourly_kwh)).

    Returns:
        List of dicts with 'hour' (index) and 'kwh' keys, sorted by kwh descending.
        Empty list when *hourly_kwh* is empty.
    """
    if not hourly_kwh:
        return []
    indexed = [{"hour": i, "kwh": round(v, 4)} for i, v in enumerate(hourly_kwh)]
    indexed.sort(key=lambda x: x["kwh"], reverse=True)
    return indexed[: min(n, len(indexed))]


def rolling_savings_summary(
    actual_series: list[float],
    baseline_series: list[float],
    window: int = 24,
) -> list[dict[str, float]]:
    """Compute rolling *window*-period savings between actual and baseline series.

    Args:
        actual_series: Measured consumption readings.
        baseline_series: Reference consumption readings (same length as actual_series).
        window: Rolling window size in periods.

    Returns:
        List of dicts with 'window_start', 'saved_kwh', 'savings_pct' for each
        complete window. Empty when inputs are too short.

    Raises:
        ValueError: If series lengths differ.
    """
    if len(actual_series) != len(baseline_series):
        raise ValueError(f"Series length mismatch: {len(actual_series)} vs {len(baseline_series)}")
    n = len(actual_series)
    if n < window:
        return []
    results = []
    for start in range(n - window + 1):
        end = start + window
        actual_window = actual_series[start:end]
        baseline_window = baseline_series[start:end]
        saved = sum(b - a for a, b in zip(actual_window, baseline_window, strict=False))
        baseline_total = sum(baseline_window)
        pct = round(100.0 * saved / baseline_total, 4) if baseline_total > 0 else 0.0
        results.append(
            {
                "window_start": float(start),
                "saved_kwh": round(saved, 4),
                "savings_pct": pct,
            }
        )
    return results


def carbon_cost_report(
    kwh_series: list[float],
    rate_per_kwh: float,
    co2_per_kwh_kg: float = 0.233,
) -> dict[str, float]:
    """Compute total energy cost and carbon impact from a kWh series.

    Args:
        kwh_series: List of energy consumption readings (kWh).
        rate_per_kwh: Tariff rate in currency per kWh.
        co2_per_kwh_kg: Grid emission factor (kg CO2 per kWh).

    Returns:
        Dict with total_kwh, total_cost, total_co2_kg, avg_kwh.
    """
    total = sum(kwh_series)
    n = len(kwh_series)
    return {
        "total_kwh": round(total, 4),
        "total_cost": round(total * rate_per_kwh, 4),
        "total_co2_kg": round(total * co2_per_kwh_kg, 4),
        "avg_kwh": round(total / n, 4) if n > 0 else 0.0,
    }


def demand_variance_report(readings: list[float]) -> dict[str, float]:
    """Summarise demand variability from a series of consumption readings.

    Args:
        readings: List of energy consumption readings.

    Returns:
        Dict with mean, std_dev, cv_pct (coefficient of variation as %),
        peak, valley, and peak_to_valley_ratio.

    Raises:
        ValueError: If *readings* is empty.
    """
    if not readings:
        raise ValueError("readings must not be empty")
    n = len(readings)
    mean = sum(readings) / n
    variance = sum((r - mean) ** 2 for r in readings) / n
    std_dev = variance ** 0.5
    peak = max(readings)
    valley = min(readings)
    cv_pct = round(std_dev / mean * 100.0, 4) if mean > 0 else 0.0
    ptv = round(peak / valley, 4) if valley > 0 else float("inf")
    return {
        "mean": round(mean, 4),
        "std_dev": round(std_dev, 4),
        "cv_pct": cv_pct,
        "peak": peak,
        "valley": valley,
        "peak_to_valley_ratio": ptv,
    }


def hourly_cost_breakdown(
    hourly_kwh: list[float],
    tariff_per_kwh: float,
) -> list[dict[str, float]]:
    """Compute per-hour energy cost from an hourly consumption series.

    Args:
        hourly_kwh: Hourly energy readings in kWh (typically 24 or 168 values).
        tariff_per_kwh: Cost per kWh.

    Returns:
        List of dicts with 'hour' (0-based index), 'kwh', and 'cost' keys.

    Raises:
        ValueError: If *hourly_kwh* is empty or *tariff_per_kwh* is negative.
    """
    if not hourly_kwh:
        raise ValueError("hourly_kwh must not be empty")
    if tariff_per_kwh < 0:
        raise ValueError("tariff_per_kwh must be non-negative")
    return [
        {"hour": i, "kwh": round(kwh, 4), "cost": round(kwh * tariff_per_kwh, 4)}
        for i, kwh in enumerate(hourly_kwh)
    ]


def savings_report(
    baseline_kwh: float,
    actual_kwh: float,
    tariff_per_kwh: float = 0.15,
) -> dict[str, float]:
    """Summarise energy savings against a baseline in kWh and monetary terms.

    Args:
        baseline_kwh: Reference (pre-improvement) energy consumption in kWh.
        actual_kwh: Measured consumption in kWh.
        tariff_per_kwh: Energy unit cost for monetary calculation.

    Returns:
        Dict with 'saved_kwh', 'saved_cost', 'pct_saved', and 'efficiency_ratio'.
        Positive 'saved_kwh' means actual < baseline (improvement).

    Raises:
        ValueError: If *tariff_per_kwh* is negative or *baseline_kwh* is negative.
    """
    if baseline_kwh < 0:
        raise ValueError("baseline_kwh must be non-negative")
    if tariff_per_kwh < 0:
        raise ValueError("tariff_per_kwh must be non-negative")
    saved_kwh = round(baseline_kwh - actual_kwh, 4)
    saved_cost = round(saved_kwh * tariff_per_kwh, 4)
    pct_saved = round(saved_kwh / baseline_kwh * 100.0, 4) if baseline_kwh > 0 else 0.0
    efficiency_ratio = round(actual_kwh / baseline_kwh, 4) if baseline_kwh > 0 else 1.0
    return {
        "saved_kwh": saved_kwh,
        "saved_cost": saved_cost,
        "pct_saved": pct_saved,
        "efficiency_ratio": efficiency_ratio,
    }
