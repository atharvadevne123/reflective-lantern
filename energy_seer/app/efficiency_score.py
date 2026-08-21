"""Building energy efficiency scoring based on consumption patterns."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

BUILDING_BENCHMARKS: dict[str, float] = {
    "residential": 4.5,
    "commercial": 12.0,
    "industrial": 30.0,
    "data_center": 60.0,
    "hospital": 25.0,
    "school": 8.0,
    "office": 10.0,
    "retail": 15.0,
}

GRADE_THRESHOLDS: dict[str, float] = {"A": 80.0, "B": 60.0, "C": 40.0}


def _grade(score: float) -> str:
    """Map a numeric efficiency score to a letter grade."""
    for letter, threshold in GRADE_THRESHOLDS.items():
        if score >= threshold:
            return letter
    return "D"


def compute_efficiency_score(
    consumption_kwh: float,
    building_type: str,
    floor_area_sqm: float = 100.0,
) -> dict[str, object]:
    """Score building energy efficiency on a 0–100 scale (higher = more efficient).

    Uses consumption per square metre compared to the benchmark for the given
    building type. Unknown building types fall back to a generic 10 kWh/sqm/day
    benchmark.

    Args:
        consumption_kwh: Daily energy consumption in kilowatt-hours.
        building_type: Building category string (case-insensitive).
        floor_area_sqm: Total conditioned floor area in square metres.

    Returns:
        Dict with 'score' (float), 'grade' (A–D), 'consumption_per_sqm',
        'benchmark_per_sqm', and 'building_type' keys.

    Raises:
        ValueError: If *consumption_kwh* is negative or *floor_area_sqm* <= 0.
    """
    if consumption_kwh < 0:
        raise ValueError(f"consumption_kwh must be non-negative, got {consumption_kwh}")
    if floor_area_sqm <= 0:
        raise ValueError(f"floor_area_sqm must be positive, got {floor_area_sqm}")
    btype = building_type.lower()
    benchmark = BUILDING_BENCHMARKS.get(btype, 10.0)
    consumption_per_sqm = consumption_kwh / floor_area_sqm
    ratio = consumption_per_sqm / max(benchmark / 100.0, 1e-6)
    score = float(np.clip(100.0 - (ratio - 1.0) * 50.0, 0.0, 100.0))
    grade = _grade(score)
    logger.debug(
        "efficiency_score: building=%s score=%.1f grade=%s ratio=%.4f",
        btype,
        score,
        grade,
        ratio,
    )
    return {
        "score": round(score, 1),
        "grade": grade,
        "consumption_per_sqm": round(consumption_per_sqm, 4),
        "benchmark_per_sqm": round(benchmark / 100.0, 4),
        "building_type": btype,
    }


def efficiency_delta(
    score_before: float,
    score_after: float,
) -> dict[str, object]:
    """Compute the change in efficiency score after an intervention.

    Args:
        score_before: Efficiency score before the intervention (0–100).
        score_after: Efficiency score after the intervention (0–100).

    Returns:
        Dict with 'delta', 'direction' ('improved'|'degraded'|'unchanged'),
        and 'grade_before' / 'grade_after' keys.
    """
    delta = round(score_after - score_before, 2)
    direction = "improved" if delta > 0 else ("degraded" if delta < 0 else "unchanged")
    return {
        "delta": delta,
        "direction": direction,
        "grade_before": _grade(score_before),
        "grade_after": _grade(score_after),
    }


def efficiency_trend(scores: list[float]) -> dict[str, object]:
    """Summarise a sequence of efficiency scores as a trend.

    Args:
        scores: Ordered list of efficiency scores (0–100).

    Returns:
        Dict with 'mean', 'min', 'max', 'trend' ('improving'|'declining'|'stable'),
        and 'latest_grade'.  Returns a zeros dict when *scores* is empty.
    """
    if not scores:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "trend": "stable", "latest_grade": "D"}
    mean_score = round(sum(scores) / len(scores), 2)
    if len(scores) >= 2:
        delta = scores[-1] - scores[0]
        trend = "improving" if delta > 1 else ("declining" if delta < -1 else "stable")
    else:
        trend = "stable"
    return {
        "mean": mean_score,
        "min": round(min(scores), 2),
        "max": round(max(scores), 2),
        "trend": trend,
        "latest_grade": _grade(scores[-1]),
    }


def normalised_efficiency_index(
    consumption_kwh: float,
    building_type: str,
    floor_area_sqm: float,
    occupancy_hours: float = 8.0,
) -> float:
    """Compute a normalised efficiency index adjusted for occupancy hours.

    Args:
        consumption_kwh: Daily energy consumption (kWh).
        building_type: Building category (case-insensitive).
        floor_area_sqm: Conditioned floor area (m²).
        occupancy_hours: Average occupied hours per day (default 8).

    Returns:
        Index in [0, 1]; higher values indicate better efficiency.
    """
    if floor_area_sqm <= 0 or occupancy_hours <= 0:
        return 0.0
    btype = building_type.lower()
    benchmark = BUILDING_BENCHMARKS.get(btype, 10.0)
    adjusted = consumption_kwh / (floor_area_sqm * (occupancy_hours / 24.0))
    index = float(np.clip(1.0 - adjusted / max(benchmark, 1e-6), 0.0, 1.0))
    return round(index, 4)


__all__ = [
    "BUILDING_BENCHMARKS",
    "GRADE_THRESHOLDS",
    "compute_efficiency_score",
    "efficiency_delta",
    "efficiency_trend",
    "normalised_efficiency_index",
]
