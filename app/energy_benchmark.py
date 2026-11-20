"""Peer benchmarking of building energy performance.

Ranks a building's Energy Use Intensity (EUI) against a peer cohort and
converts that rank into the 1-100 score and letter grade facilities teams
report upward. Complements ``app.benchmarks``, which holds the static
reference EUI tables, by scoring against an actual observed cohort.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MIN_COHORT_SIZE: int = 3
"""Below this, a percentile rank is too noisy to be worth reporting."""

GRADE_THRESHOLDS: tuple[tuple[int, str], ...] = (
    (90, "A"),
    (75, "B"),
    (50, "C"),
    (25, "D"),
    (0, "F"),
)
"""Score floors mapped to letter grades, highest first."""


@dataclass
class BenchmarkResult:
    """A building's standing within its peer cohort."""

    eui: float
    cohort_size: int
    cohort_median_eui: float
    percentile_rank: float
    score: int
    grade: str
    savings_potential_kwh: float


def energy_use_intensity(annual_kwh: float, floor_area_m2: float) -> float:
    """Return annual energy use per square metre.

    Args:
        annual_kwh: Total annual consumption in kWh.
        floor_area_m2: Gross internal floor area in square metres.

    Returns:
        EUI in kWh/m2 rounded to 4 decimal places.

    Raises:
        ValueError: If *annual_kwh* is negative or *floor_area_m2* is not
            positive.
    """
    if annual_kwh < 0:
        raise ValueError(f"annual_kwh must be non-negative, got {annual_kwh}")
    if floor_area_m2 <= 0:
        raise ValueError(f"floor_area_m2 must be positive, got {floor_area_m2}")
    return round(annual_kwh / floor_area_m2, 4)


def percentile_rank(eui: float, cohort_euis: list[float]) -> float:
    """Return the building's percentile standing, where higher is better.

    Because lower energy use is better, the rank counts the share of the
    cohort the building beats — a building using less than everyone else
    scores 100.

    Args:
        eui: The building's energy use intensity.
        cohort_euis: EUI values for the peer cohort.

    Returns:
        Percentile in 0-100 rounded to 2 decimal places.

    Raises:
        ValueError: If *cohort_euis* is empty.
    """
    if not cohort_euis:
        raise ValueError("cohort_euis must not be empty")
    beaten = sum(1 for peer in cohort_euis if eui < peer)
    tied = sum(1 for peer in cohort_euis if eui == peer)
    # Ties split the difference, so identical buildings share a rank.
    return round(100.0 * (beaten + 0.5 * tied) / len(cohort_euis), 2)


def score_from_percentile(percentile: float) -> int:
    """Convert a percentile rank into an integer 1-100 score.

    Args:
        percentile: Percentile rank in 0-100.

    Returns:
        Score clamped to 1-100.

    Raises:
        ValueError: If *percentile* is outside 0-100.
    """
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in 0-100, got {percentile}")
    return max(1, min(100, round(percentile)))


def grade_from_score(score: int) -> str:
    """Map a 1-100 score onto a letter grade.

    Args:
        score: Performance score in 1-100.

    Returns:
        Letter grade from ``"A"`` down to ``"F"``.
    """
    for floor, grade in GRADE_THRESHOLDS:
        if score >= floor:
            return grade
    return "F"


def savings_potential(annual_kwh: float, eui: float, target_eui: float) -> float:
    """Return the annual kWh saved by reaching a target EUI.

    Args:
        annual_kwh: Current annual consumption in kWh.
        eui: The building's current energy use intensity.
        target_eui: The EUI being aimed for.

    Returns:
        Savings in kWh rounded to 3 decimal places. Zero when the building
        already meets or beats the target.

    Raises:
        ValueError: If *eui* is not positive or *target_eui* is negative.
    """
    if eui <= 0:
        raise ValueError(f"eui must be positive, got {eui}")
    if target_eui < 0:
        raise ValueError(f"target_eui must be non-negative, got {target_eui}")
    if target_eui >= eui:
        return 0.0
    return round(annual_kwh * (1.0 - target_eui / eui), 3)


def benchmark(
    annual_kwh: float,
    floor_area_m2: float,
    cohort_euis: list[float],
) -> BenchmarkResult:
    """Benchmark a building against its peer cohort.

    Savings potential is measured against the cohort median, which is the
    conventional "reach the middle of your peers" target.

    Args:
        annual_kwh: Total annual consumption in kWh.
        floor_area_m2: Gross internal floor area in square metres.
        cohort_euis: EUI values for the peer cohort.

    Returns:
        A populated :class:`BenchmarkResult`.

    Raises:
        ValueError: If the cohort is smaller than :data:`MIN_COHORT_SIZE`,
            or any consumption or area value is invalid.
    """
    if len(cohort_euis) < MIN_COHORT_SIZE:
        raise ValueError(
            f"cohort must hold at least {MIN_COHORT_SIZE} peers for a meaningful rank, got {len(cohort_euis)}"
        )

    eui = energy_use_intensity(annual_kwh, floor_area_m2)
    percentile = percentile_rank(eui, cohort_euis)
    score = score_from_percentile(percentile)
    median = round(statistics.median(cohort_euis), 4)

    result = BenchmarkResult(
        eui=eui,
        cohort_size=len(cohort_euis),
        cohort_median_eui=median,
        percentile_rank=percentile,
        score=score,
        grade=grade_from_score(score),
        savings_potential_kwh=savings_potential(annual_kwh, eui, median),
    )
    logger.info(
        "Benchmark: EUI %.2f kWh/m2, %.1f percentile of %d peers, grade %s",
        eui,
        percentile,
        len(cohort_euis),
        result.grade,
    )
    return result


__all__ = [
    "GRADE_THRESHOLDS",
    "MIN_COHORT_SIZE",
    "BenchmarkResult",
    "benchmark",
    "energy_use_intensity",
    "grade_from_score",
    "percentile_rank",
    "savings_potential",
    "score_from_percentile",
]
