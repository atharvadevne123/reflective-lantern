"""Battery energy storage degradation and lifetime modelling."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

NOMINAL_CYCLES_LFP: int = 3000
"""Typical cycle life for LFP (LiFePO4) cells at 80% DoD."""

NOMINAL_CYCLES_NMC: int = 1500
"""Typical cycle life for NMC cells at 80% DoD."""

NOMINAL_CYCLES_LEAD_ACID: int = 600
"""Typical cycle life for lead-acid at 50% DoD."""

EOL_CAPACITY_FRACTION: float = 0.80
"""End-of-life capacity threshold (80% of nameplate)."""

CALENDAR_DEGRADATION_PER_YEAR: float = 0.02
"""Approximate calendar degradation per year at room temperature (2%)."""


@dataclass
class DegradationResult:
    """Battery degradation state at a given point in life."""

    cycles_completed: float
    calendar_years: float
    capacity_fraction: float
    capacity_kwh: float
    cycle_degradation: float
    calendar_degradation: float
    total_degradation: float
    eol_reached: bool
    remaining_cycles: float


def rainflow_cycle_depth_factor(depth_of_discharge: float) -> float:
    """Cycle depth correction factor using simplified Rainflow weighting.

    A full cycle at partial DoD counts as a fraction of a full-DoD cycle.

    Args:
        depth_of_discharge: Depth of discharge as a fraction in (0, 1].

    Returns:
        Effective cycle weight factor.

    Raises:
        ValueError: If DoD is outside (0, 1].
    """
    if not (0 < depth_of_discharge <= 1):
        raise ValueError("depth_of_discharge must be in (0, 1]")
    return round(depth_of_discharge ** 1.5, 6)


def cycle_degradation(
    cycles: float,
    nominal_cycles: int,
    depth_of_discharge: float = 0.80,
) -> float:
    """Capacity loss due to cycling.

    Uses a linear degradation model between 0 and EOL.

    Args:
        cycles: Number of full-equivalent cycles completed.
        nominal_cycles: Rated cycle life at the reference DoD.
        depth_of_discharge: Operating depth of discharge (fraction).

    Returns:
        Capacity degradation as a fraction [0, 1].

    Raises:
        ValueError: If cycles < 0 or nominal_cycles <= 0.
    """
    if cycles < 0:
        raise ValueError("cycles must be non-negative")
    if nominal_cycles <= 0:
        raise ValueError("nominal_cycles must be positive")
    dod_factor = rainflow_cycle_depth_factor(depth_of_discharge)
    effective_cycles = cycles * dod_factor
    loss_per_cycle = (1.0 - EOL_CAPACITY_FRACTION) / nominal_cycles
    return round(min(1.0, effective_cycles * loss_per_cycle), 6)


def calendar_degradation(
    years: float,
    degradation_per_year: float = CALENDAR_DEGRADATION_PER_YEAR,
    temperature_factor: float = 1.0,
) -> float:
    """Capacity loss due to calendar ageing.

    Args:
        years: Elapsed time in years.
        degradation_per_year: Annual capacity loss fraction at reference temperature.
        temperature_factor: Multiplier for elevated temperature (>1 = worse).

    Returns:
        Capacity degradation as a fraction [0, 1].

    Raises:
        ValueError: If years < 0 or degradation_per_year <= 0.
    """
    if years < 0:
        raise ValueError("years must be non-negative")
    if degradation_per_year <= 0:
        raise ValueError("degradation_per_year must be positive")
    return round(min(1.0, years * degradation_per_year * temperature_factor), 6)


def remaining_capacity(
    nameplate_kwh: float,
    cycle_deg: float,
    calendar_deg: float,
) -> float:
    """Compute current usable capacity after degradation.

    Args:
        nameplate_kwh: Original nameplate capacity in kWh.
        cycle_deg: Cycle degradation fraction.
        calendar_deg: Calendar degradation fraction.

    Returns:
        Current capacity in kWh (not below zero).

    Raises:
        ValueError: If nameplate_kwh <= 0 or degradation fractions are out of [0, 1].
    """
    if nameplate_kwh <= 0:
        raise ValueError("nameplate_kwh must be positive")
    for d, name in [(cycle_deg, "cycle_deg"), (calendar_deg, "calendar_deg")]:
        if not (0 <= d <= 1):
            raise ValueError(f"{name} must be in [0, 1]")
    total_deg = min(1.0, cycle_deg + calendar_deg)
    return round(max(0.0, nameplate_kwh * (1.0 - total_deg)), 4)


def cycles_to_eol(
    nominal_cycles: int,
    current_cycles: float,
    current_calendar_years: float,
    depth_of_discharge: float = 0.80,
    degradation_per_year: float = CALENDAR_DEGRADATION_PER_YEAR,
) -> float:
    """Estimate remaining cycles until end-of-life.

    Args:
        nominal_cycles: Rated cycle life at reference DoD.
        current_cycles: Cycles already completed.
        current_calendar_years: Calendar age in years so far.
        depth_of_discharge: Operating DoD.
        degradation_per_year: Calendar degradation rate per year.

    Returns:
        Estimated remaining full-equivalent cycles.
    """
    cyc_deg = cycle_degradation(current_cycles, nominal_cycles, depth_of_discharge)
    cal_deg = calendar_degradation(current_calendar_years, degradation_per_year)
    total_deg = cyc_deg + cal_deg
    remaining_cap_frac = max(0.0, 1.0 - total_deg)
    if remaining_cap_frac <= EOL_CAPACITY_FRACTION:
        return 0.0
    remaining_budget = remaining_cap_frac - EOL_CAPACITY_FRACTION
    dod_factor = rainflow_cycle_depth_factor(depth_of_discharge)
    loss_per_cycle = (1.0 - EOL_CAPACITY_FRACTION) / nominal_cycles
    if loss_per_cycle == 0:
        return float("inf")
    return round(remaining_budget / (loss_per_cycle * dod_factor), 1)


def degradation_report(
    nameplate_kwh: float,
    cycles_completed: float,
    calendar_years: float,
    nominal_cycles: int = NOMINAL_CYCLES_LFP,
    depth_of_discharge: float = 0.80,
    degradation_per_year: float = CALENDAR_DEGRADATION_PER_YEAR,
) -> DegradationResult:
    """Generate a comprehensive battery degradation report.

    Args:
        nameplate_kwh: Original nameplate capacity in kWh.
        cycles_completed: Full-equivalent cycles completed.
        calendar_years: Elapsed calendar years.
        nominal_cycles: Rated cycle life at reference DoD.
        depth_of_discharge: Operating depth of discharge.
        degradation_per_year: Calendar ageing rate per year.

    Returns:
        Populated :class:`DegradationResult`.
    """
    cyc_deg = cycle_degradation(cycles_completed, nominal_cycles, depth_of_discharge)
    cal_deg = calendar_degradation(calendar_years, degradation_per_year)
    total_deg = min(1.0, cyc_deg + cal_deg)
    cap_frac = max(0.0, 1.0 - total_deg)
    cap_kwh = remaining_capacity(nameplate_kwh, cyc_deg, cal_deg)
    eol = cap_frac <= EOL_CAPACITY_FRACTION
    rem_cycles = cycles_to_eol(nominal_cycles, cycles_completed, calendar_years, depth_of_discharge, degradation_per_year)

    return DegradationResult(
        cycles_completed=cycles_completed,
        calendar_years=calendar_years,
        capacity_fraction=round(cap_frac, 6),
        capacity_kwh=cap_kwh,
        cycle_degradation=cyc_deg,
        calendar_degradation=cal_deg,
        total_degradation=round(total_deg, 6),
        eol_reached=eol,
        remaining_cycles=rem_cycles,
    )


def state_of_health(
    current_capacity_kwh: float,
    initial_capacity_kwh: float,
) -> float:
    """Return the State of Health (SoH) as a fraction in [0, 1].

    Args:
        current_capacity_kwh: Current measurable capacity in kWh.
        initial_capacity_kwh: Nominal initial capacity in kWh (must be positive).

    Returns:
        SoH fraction (1.0 = new, 0.8 = end of life for most batteries).

    Raises:
        ValueError: If initial_capacity_kwh ≤ 0 or current_capacity_kwh < 0.
    """
    if initial_capacity_kwh <= 0:
        raise ValueError("initial_capacity_kwh must be positive")
    if current_capacity_kwh < 0:
        raise ValueError("current_capacity_kwh must be non-negative")
    return min(current_capacity_kwh / initial_capacity_kwh, 1.0)


def energy_throughput_kwh(
    cycles: float,
    capacity_kwh: float,
    depth_of_discharge: float = 0.8,
) -> float:
    """Return total energy throughput in kWh for a given number of cycles.

    Args:
        cycles: Number of charge-discharge cycles.
        capacity_kwh: Battery usable capacity in kWh (must be positive).
        depth_of_discharge: Fraction of capacity used per cycle (0 < DoD ≤ 1).

    Returns:
        Total energy throughput in kWh.

    Raises:
        ValueError: If cycles < 0, capacity_kwh ≤ 0, or DoD invalid.
    """
    if cycles < 0:
        raise ValueError("cycles must be non-negative")
    if capacity_kwh <= 0:
        raise ValueError("capacity_kwh must be positive")
    if not 0 < depth_of_discharge <= 1.0:
        raise ValueError("depth_of_discharge must be in (0, 1]")
    return cycles * capacity_kwh * depth_of_discharge


def replacement_cost_per_kwh(
    pack_cost: float,
    capacity_kwh: float,
) -> float:
    """Return battery replacement cost per kWh of capacity.

    Args:
        pack_cost: Total pack replacement cost in currency units.
        capacity_kwh: Battery capacity in kWh (must be positive).

    Returns:
        Cost per kWh in currency units/kWh.

    Raises:
        ValueError: If pack_cost < 0 or capacity_kwh ≤ 0.
    """
    if pack_cost < 0:
        raise ValueError("pack_cost must be non-negative")
    if capacity_kwh <= 0:
        raise ValueError("capacity_kwh must be positive")
    return pack_cost / capacity_kwh
