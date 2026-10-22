"""Stochastic occupancy modelling for commercial and residential buildings."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OccupancyProfile:
    """Predicted hourly occupancy fractions for a 24-hour period."""

    hourly_fractions: list[float]
    peak_hour: int
    peak_fraction: float
    average_fraction: float
    occupied_hours: int


OFFICE_BASE_SCHEDULE: dict[int, float] = {
    0: 0.0,
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 0.0,
    5: 0.0,
    6: 0.05,
    7: 0.20,
    8: 0.60,
    9: 0.90,
    10: 0.95,
    11: 0.95,
    12: 0.80,
    13: 0.75,
    14: 0.90,
    15: 0.90,
    16: 0.85,
    17: 0.60,
    18: 0.30,
    19: 0.10,
    20: 0.05,
    21: 0.0,
    22: 0.0,
    23: 0.0,
}

RESIDENTIAL_BASE_SCHEDULE: dict[int, float] = {
    0: 0.95,
    1: 0.95,
    2: 0.95,
    3: 0.95,
    4: 0.90,
    5: 0.80,
    6: 0.70,
    7: 0.60,
    8: 0.40,
    9: 0.35,
    10: 0.30,
    11: 0.30,
    12: 0.40,
    13: 0.35,
    14: 0.30,
    15: 0.35,
    16: 0.50,
    17: 0.65,
    18: 0.80,
    19: 0.85,
    20: 0.90,
    21: 0.90,
    22: 0.90,
    23: 0.95,
}


def apply_daytype_factor(
    base_schedule: dict[int, float],
    is_weekend: bool,
) -> dict[int, float]:
    """Scale occupancy fractions for weekend vs weekday.

    Args:
        base_schedule: Mapping from hour (0-23) to weekday occupancy fraction.
        is_weekend: True scales the schedule down (offices empty, homes fuller).

    Returns:
        Adjusted schedule with the same keys.
    """
    if is_weekend:
        return {h: min(1.0, f * 0.15) for h, f in base_schedule.items()}
    return dict(base_schedule)


def build_occupancy_profile(
    schedule: dict[int, float] | None = None,
    is_weekend: bool = False,
    occupied_threshold: float = 0.05,
) -> OccupancyProfile:
    """Build a 24-hour occupancy profile from a base schedule.

    Args:
        schedule: Hour-to-fraction mapping; defaults to office schedule.
        is_weekend: Whether to apply weekend scaling.
        occupied_threshold: Fraction above which an hour is considered occupied.

    Returns:
        Populated :class:`OccupancyProfile`.
    """
    if schedule is None:
        schedule = OFFICE_BASE_SCHEDULE
    adjusted = apply_daytype_factor(schedule, is_weekend)
    hourly = [adjusted.get(h, 0.0) for h in range(24)]
    peak_hour = int(max(range(24), key=lambda h: hourly[h]))
    peak_frac = hourly[peak_hour]
    avg_frac = round(sum(hourly) / 24.0, 4)
    occupied_hours = sum(1 for f in hourly if f >= occupied_threshold)
    return OccupancyProfile(
        hourly_fractions=hourly,
        peak_hour=peak_hour,
        peak_fraction=peak_frac,
        average_fraction=avg_frac,
        occupied_hours=occupied_hours,
    )


def occupancy_adjusted_setpoint(
    occupied_setpoint_c: float,
    unoccupied_setpoint_c: float,
    occupancy_fraction: float,
    transition_threshold: float = 0.05,
) -> float:
    """Derive the effective HVAC setpoint for a given occupancy fraction.

    Args:
        occupied_setpoint_c: Setpoint when the space is occupied (°C).
        unoccupied_setpoint_c: Setpoint when unoccupied (°C).
        occupancy_fraction: Current occupancy level in [0, 1].
        transition_threshold: Fraction above which the occupied setpoint applies.

    Returns:
        Effective temperature setpoint in °C.

    Raises:
        ValueError: If occupancy_fraction is outside [0, 1].
    """
    if not (0.0 <= occupancy_fraction <= 1.0):
        raise ValueError("occupancy_fraction must be in [0, 1]")
    if occupancy_fraction >= transition_threshold:
        return occupied_setpoint_c
    return unoccupied_setpoint_c


def energy_with_occupancy(
    base_hourly_kwh: list[float],
    hourly_fractions: list[float],
    occupancy_reduction_factor: float = 0.30,
) -> list[float]:
    """Scale hourly energy consumption by occupancy fraction.

    Models the savings from setback controls that reduce consumption during
    unoccupied periods.

    Args:
        base_hourly_kwh: Full-occupancy energy consumption per hour.
        hourly_fractions: Occupancy fractions per hour (each in [0, 1]).
        occupancy_reduction_factor: Maximum fractional reduction at zero occupancy.

    Returns:
        Adjusted energy series.

    Raises:
        ValueError: If lists differ in length or reduction_factor outside [0, 1).
    """
    if len(base_hourly_kwh) != len(hourly_fractions):
        raise ValueError("base_hourly_kwh and hourly_fractions must have the same length")
    if not (0.0 <= occupancy_reduction_factor < 1.0):
        raise ValueError("occupancy_reduction_factor must be in [0, 1)")
    result: list[float] = []
    for kwh, frac in zip(base_hourly_kwh, hourly_fractions, strict=False):
        frac = max(0.0, min(1.0, frac))
        scale = 1.0 - occupancy_reduction_factor * (1.0 - frac)
        result.append(round(kwh * scale, 6))
    return result


def annual_occupied_hours(
    weekday_schedule: dict[int, float],
    weekend_schedule: dict[int, float] | None = None,
    threshold: float = 0.05,
    weekdays_per_year: int = 261,
) -> int:
    """Estimate total occupied hours per year from weekday and weekend schedules.

    Args:
        weekday_schedule: Weekday hour-to-fraction mapping.
        weekend_schedule: Weekend schedule; defaults to an empty building.
        threshold: Fraction above which the hour is considered occupied.
        weekdays_per_year: Number of weekdays in the year.

    Returns:
        Estimated occupied hours per year.
    """
    if weekend_schedule is None:
        weekend_schedule = {h: 0.0 for h in range(24)}
    weekday_hours = sum(1 for f in weekday_schedule.values() if f >= threshold)
    weekend_hours = sum(1 for f in weekend_schedule.values() if f >= threshold)
    weekend_days = 365 - weekdays_per_year
    return weekday_hours * weekdays_per_year + weekend_hours * weekend_days
