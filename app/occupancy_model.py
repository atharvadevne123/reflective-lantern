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
    name: str = ""


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
    is_weekend: bool = False,
    *,
    factor: float | None = None,
) -> dict[int, float]:
    """Scale occupancy fractions for weekend vs weekday.

    Args:
        base_schedule: Mapping from hour (0-23) to weekday occupancy fraction.
        is_weekend: True scales the schedule down (offices empty, homes fuller).
        factor: Custom scaling factor; overrides is_weekend logic when provided.

    Returns:
        Adjusted schedule with the same keys.
    """
    if factor is not None:
        return {h: min(1.0, f * factor) for h, f in base_schedule.items()}
    if is_weekend:
        return {h: min(1.0, f * 0.15) for h, f in base_schedule.items()}
    return dict(base_schedule)


_NAMED_SCHEDULES: dict[str, dict[int, float]] = {}


def _get_named_schedules() -> dict[str, dict[int, float]]:
    if not _NAMED_SCHEDULES:
        _NAMED_SCHEDULES["office"] = OFFICE_BASE_SCHEDULE
        _NAMED_SCHEDULES["residential"] = RESIDENTIAL_BASE_SCHEDULE
    return _NAMED_SCHEDULES


def build_occupancy_profile(
    schedule: dict[int, float] | str | None = None,
    is_weekend: bool = False,
    occupied_threshold: float = 0.05,
) -> OccupancyProfile:
    """Build a 24-hour occupancy profile from a base schedule.

    Args:
        schedule: Hour-to-fraction mapping, a named profile ("office", "residential"),
            or None to default to the office schedule.
        is_weekend: Whether to apply weekend scaling.
        occupied_threshold: Fraction above which an hour is considered occupied.

    Returns:
        Populated :class:`OccupancyProfile`.
    """
    profile_name = ""
    if schedule is None:
        schedule = OFFICE_BASE_SCHEDULE
        profile_name = "office"
    elif isinstance(schedule, str):
        profile_name = schedule
        named = _get_named_schedules()
        if schedule not in named:
            raise ValueError(f"Unknown schedule name: {schedule!r}. Use one of {list(named)}")
        schedule = named[schedule]
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
        name=profile_name,
    )


def occupancy_adjusted_setpoint(
    occupied_setpoint_c: float,
    unoccupied_setpoint_c_or_fraction: float | None = None,
    occupancy_fraction: float | None = None,
    transition_threshold: float = 0.05,
    *,
    occupied: bool | None = None,
    unoccupied_offset_c: float = 2.0,
) -> float:
    """Derive the effective HVAC setpoint for a given occupancy fraction.

    Accepts two calling conventions:
    - (occupied_setpoint_c, unoccupied_setpoint_c, occupancy_fraction): original
    - (setpoint_c, occupied=True/False): simplified boolean occupied flag

    Args:
        occupied_setpoint_c: Setpoint when the space is occupied (°C).
        unoccupied_setpoint_c_or_fraction: Unoccupied setpoint or occupancy fraction.
        occupancy_fraction: Occupancy fraction in [0, 1] (original convention).
        transition_threshold: Fraction above which the occupied setpoint applies.
        occupied: Boolean override — when provided, returns occupied or unoccupied setpoint.
        unoccupied_offset_c: Offset subtracted from setpoint for unoccupied mode.

    Returns:
        Effective temperature setpoint in °C.

    Raises:
        ValueError: If occupancy_fraction is outside [0, 1].
    """
    if occupied is not None:
        return occupied_setpoint_c if occupied else occupied_setpoint_c - unoccupied_offset_c
    if occupancy_fraction is None and unoccupied_setpoint_c_or_fraction is not None:
        # Only 2 args: (occupied_setpoint, occupancy_fraction)
        frac = float(unoccupied_setpoint_c_or_fraction)
        if not (0.0 <= frac <= 1.0):
            raise ValueError("occupancy_fraction must be in [0, 1]")
        return occupied_setpoint_c if frac >= transition_threshold else occupied_setpoint_c - unoccupied_offset_c
    unoccupied_setpoint_c = unoccupied_setpoint_c_or_fraction if unoccupied_setpoint_c_or_fraction is not None else occupied_setpoint_c - unoccupied_offset_c
    frac = occupancy_fraction if occupancy_fraction is not None else 1.0
    if not (0.0 <= frac <= 1.0):
        raise ValueError("occupancy_fraction must be in [0, 1]")
    if frac >= transition_threshold:
        return occupied_setpoint_c
    return unoccupied_setpoint_c


def energy_with_occupancy(
    base_hourly_kwh: list[float] | float,
    hourly_fractions: list[float] | dict[int, float],
    occupancy_reduction_factor: float = 0.30,
) -> list[float] | float:
    """Scale hourly energy consumption by occupancy fraction.

    Accepts two calling conventions:
    - (list_of_kwh, list_of_fractions): original; returns adjusted list
    - (base_total_kwh, dict_of_fractions): simplified; returns total adjusted float

    Args:
        base_hourly_kwh: Full-occupancy energy per hour (list) or total base kWh (float).
        hourly_fractions: Occupancy fractions as list or {hour: fraction} dict.
        occupancy_reduction_factor: Maximum fractional reduction at zero occupancy.

    Returns:
        Adjusted energy series or total.

    Raises:
        ValueError: If lists differ in length or reduction_factor outside [0, 1).
    """
    if not (0.0 <= occupancy_reduction_factor < 1.0):
        raise ValueError("occupancy_reduction_factor must be in [0, 1)")
    if isinstance(hourly_fractions, dict):
        fracs = [hourly_fractions.get(h, 0.0) for h in range(24)]
    else:
        fracs = hourly_fractions
    if isinstance(base_hourly_kwh, (int, float)):
        base_per_hour = float(base_hourly_kwh) / 24.0
        total = 0.0
        for frac in fracs:
            frac = max(0.0, min(1.0, frac))
            scale = 1.0 - occupancy_reduction_factor * (1.0 - frac)
            total += base_per_hour * scale
        return round(total, 4)
    if len(base_hourly_kwh) != len(fracs):
        raise ValueError("base_hourly_kwh and hourly_fractions must have the same length")
    result: list[float] = []
    for kwh, frac in zip(base_hourly_kwh, fracs, strict=False):
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
