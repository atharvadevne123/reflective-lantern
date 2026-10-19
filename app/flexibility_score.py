"""Demand flexibility scoring for smart grid and demand-response programs."""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MIN_SCORE: float = 0.0
MAX_SCORE: float = 100.0


@dataclass
class FlexibilityAssessment:
    """Demand flexibility assessment result for a building or asset."""

    asset_id: str
    flexibility_score: float
    shiftable_kwh: float
    controllable_kw: float
    response_time_minutes: int
    recommendation: str


def _clamp(value: float, low: float = MIN_SCORE, high: float = MAX_SCORE) -> float:
    return max(low, min(high, value))


def load_variability_score(hourly_kwh: list[float]) -> float:
    """Score how variable a load profile is (higher = more shiftable).

    Args:
        hourly_kwh: 24-hour or similar interval readings.

    Returns:
        Variability score in [0, 100].

    Raises:
        ValueError: If fewer than 2 readings are provided.
    """
    if len(hourly_kwh) < 2:
        raise ValueError("Need at least 2 readings for variability score")
    mean = statistics.mean(hourly_kwh)
    if mean == 0.0:
        return 0.0
    cv = statistics.pstdev(hourly_kwh) / mean
    return round(_clamp(cv * 100.0), 2)


def response_readiness_score(response_time_minutes: int) -> float:
    """Score how quickly a load can respond to a curtailment signal.

    Full score (100) for instant response; zero for ≥ 60 minutes.

    Args:
        response_time_minutes: Estimated response latency in minutes.

    Returns:
        Readiness score in [0, 100].

    Raises:
        ValueError: If response_time_minutes is negative.
    """
    if response_time_minutes < 0:
        raise ValueError("response_time_minutes must be non-negative")
    if response_time_minutes == 0:
        return 100.0
    score = max(0.0, 100.0 - (response_time_minutes / 60.0) * 100.0)
    return round(score, 2)


def peak_contribution_score(
    hourly_kwh: list[float],
    system_peak_hour: int = 18,
) -> float:
    """Score the load's contribution to system peak demand.

    Higher score means the load consumes more during the system peak window,
    which indicates greater potential value in a demand-response programme.

    Args:
        hourly_kwh: 24-hour consumption profile.
        system_peak_hour: Hour-of-day at which the system peaks (0-23).

    Returns:
        Peak contribution score in [0, 100].

    Raises:
        ValueError: If hourly_kwh does not have exactly 24 entries.
    """
    if len(hourly_kwh) != 24:
        raise ValueError("hourly_kwh must have 24 entries")
    peak_window = slice(max(0, system_peak_hour - 2), min(24, system_peak_hour + 3))
    peak_sum = sum(hourly_kwh[peak_window])
    total = sum(hourly_kwh)
    if total == 0.0:
        return 0.0
    return round(_clamp((peak_sum / total) * 100.0), 2)


def overall_flexibility_score(
    variability_score: float,
    readiness_score: float,
    peak_contribution: float,
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> float:
    """Compute a weighted composite flexibility score.

    Args:
        variability_score: Output of :func:`load_variability_score`.
        readiness_score: Output of :func:`response_readiness_score`.
        peak_contribution: Output of :func:`peak_contribution_score`.
        weights: (variability_weight, readiness_weight, peak_weight).
            Must sum to 1.0.

    Returns:
        Composite score in [0, 100].

    Raises:
        ValueError: If weights do not sum to approximately 1.0.
    """
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1.0")
    score = (
        weights[0] * variability_score
        + weights[1] * readiness_score
        + weights[2] * peak_contribution
    )
    return round(_clamp(score), 2)


def assess_flexibility(
    asset_id: str,
    hourly_kwh: list[float],
    response_time_minutes: int,
    shiftable_kwh: float,
    controllable_kw: float,
) -> FlexibilityAssessment:
    """Build a complete :class:`FlexibilityAssessment` for an asset.

    Args:
        asset_id: Unique identifier for the building or asset.
        hourly_kwh: 24-hour consumption profile.
        response_time_minutes: Curtailment response latency in minutes.
        shiftable_kwh: Estimated shiftable energy in kWh.
        controllable_kw: Controllable load capacity in kW.

    Returns:
        Populated :class:`FlexibilityAssessment`.
    """
    var_score = load_variability_score(hourly_kwh)
    ready_score = response_readiness_score(response_time_minutes)
    peak_score = peak_contribution_score(hourly_kwh)
    composite = overall_flexibility_score(var_score, ready_score, peak_score)
    if composite >= 70:
        rec = "High flexibility — enrol in active demand-response programme"
    elif composite >= 40:
        rec = "Moderate flexibility — suitable for automated demand shifting"
    else:
        rec = "Low flexibility — focus on efficiency improvements first"
    return FlexibilityAssessment(
        asset_id=asset_id,
        flexibility_score=composite,
        shiftable_kwh=shiftable_kwh,
        controllable_kw=controllable_kw,
        response_time_minutes=response_time_minutes,
        recommendation=rec,
    )
