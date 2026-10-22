"""Battery energy storage system modelling and dispatch optimisation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class BatteryState(StrEnum):
    """Operating state of a battery system."""

    CHARGING = "charging"
    DISCHARGING = "discharging"
    IDLE = "idle"


@dataclass
class BatterySystem:
    """Specification of a battery energy storage system."""

    capacity_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    round_trip_efficiency: float = 0.90
    initial_soc_pct: float = 50.0


@dataclass
class DispatchResult:
    """Outcome of a battery dispatch simulation."""

    final_soc_pct: float
    energy_charged_kwh: float
    energy_discharged_kwh: float
    net_grid_kwh: float
    peak_shaved_kw: float
    cycle_count: float
    cost_saving: float


def usable_energy_kwh(
    battery: BatterySystem,
    soc_pct: float,
) -> float:
    """Compute usable energy available at a given state of charge.

    Args:
        battery: Battery system specification.
        soc_pct: Current state of charge in %.

    Returns:
        Usable energy in kWh.

    Raises:
        ValueError: If soc_pct is outside [0, 100].
    """
    if not (0 <= soc_pct <= 100):
        raise ValueError("soc_pct must be in [0, 100]")
    return round(battery.capacity_kwh * soc_pct / 100.0, 4)


def charge_kwh(
    battery: BatterySystem,
    power_kw: float,
    hours: float,
    current_soc_pct: float,
) -> tuple[float, float]:
    """Charge battery for a given period and return energy drawn and new SOC.

    Args:
        battery: Battery system specification.
        power_kw: Charging power in kW (positive).
        hours: Duration of charging in hours.
        current_soc_pct: Current state of charge in %.

    Returns:
        Tuple of (grid_energy_kwh, new_soc_pct).

    Raises:
        ValueError: If power_kw or hours are non-positive, or soc out of range.
    """
    if power_kw <= 0:
        raise ValueError("power_kw must be positive")
    if hours <= 0:
        raise ValueError("hours must be positive")
    if not (0 <= current_soc_pct <= 100):
        raise ValueError("current_soc_pct must be in [0, 100]")
    effective_power = min(power_kw, battery.max_charge_kw)
    grid_energy = effective_power * hours
    energy_stored = grid_energy * battery.round_trip_efficiency
    current_energy = usable_energy_kwh(battery, current_soc_pct)
    new_energy = min(battery.capacity_kwh, current_energy + energy_stored)
    new_soc = round(new_energy / battery.capacity_kwh * 100.0, 2)
    actual_grid = round((new_energy - current_energy) / battery.round_trip_efficiency, 4)
    return actual_grid, new_soc


def discharge_kwh(
    battery: BatterySystem,
    power_kw: float,
    hours: float,
    current_soc_pct: float,
) -> tuple[float, float]:
    """Discharge battery for a given period and return energy delivered and new SOC.

    Args:
        battery: Battery system specification.
        power_kw: Discharge power in kW (positive).
        hours: Duration in hours.
        current_soc_pct: Current state of charge in %.

    Returns:
        Tuple of (energy_delivered_kwh, new_soc_pct).

    Raises:
        ValueError: If arguments are invalid.
    """
    if power_kw <= 0:
        raise ValueError("power_kw must be positive")
    if hours <= 0:
        raise ValueError("hours must be positive")
    if not (0 <= current_soc_pct <= 100):
        raise ValueError("current_soc_pct must be in [0, 100]")
    effective_power = min(power_kw, battery.max_discharge_kw)
    requested = effective_power * hours
    available = usable_energy_kwh(battery, current_soc_pct)
    delivered = min(requested, available)
    new_energy = max(0.0, available - delivered)
    new_soc = round(new_energy / battery.capacity_kwh * 100.0, 2)
    return round(delivered, 4), new_soc


def peak_shaving_threshold(
    load_profile_kw: list[float],
    target_reduction_pct: float,
) -> float:
    """Compute a demand threshold for peak shaving.

    Args:
        load_profile_kw: Hourly load values in kW.
        target_reduction_pct: Desired peak reduction as a percentage (0–100).

    Returns:
        Demand threshold in kW at which to cap load.

    Raises:
        ValueError: If load_profile is empty or target is out of range.
    """
    if not load_profile_kw:
        raise ValueError("load_profile_kw must not be empty")
    if not (0 <= target_reduction_pct <= 100):
        raise ValueError("target_reduction_pct must be in [0, 100]")
    peak = max(load_profile_kw)
    return round(peak * (1.0 - target_reduction_pct / 100.0), 4)


def simulate_peak_shaving(
    battery: BatterySystem,
    load_profile_kw: list[float],
    demand_threshold_kw: float,
    demand_charge_per_kw: float = 15.0,
) -> DispatchResult:
    """Simulate battery dispatch for peak shaving over a load profile.

    Args:
        battery: Battery system specification.
        load_profile_kw: Hourly load values in kW.
        demand_threshold_kw: Demand threshold above which to discharge.
        demand_charge_per_kw: Monthly demand charge in $/kW.

    Returns:
        Populated :class:`DispatchResult`.

    Raises:
        ValueError: If load_profile is empty or demand_charge is negative.
    """
    if not load_profile_kw:
        raise ValueError("load_profile_kw must not be empty")
    if demand_charge_per_kw < 0:
        raise ValueError("demand_charge_per_kw must be non-negative")
    soc = battery.initial_soc_pct
    total_charged = 0.0
    total_discharged = 0.0
    net_grid = 0.0
    original_peak = max(load_profile_kw)
    effective_peak = original_peak

    for load in load_profile_kw:
        if load > demand_threshold_kw:
            needed = load - demand_threshold_kw
            delivered, soc = discharge_kwh(battery, needed, 1.0, soc)
            total_discharged += delivered
            net_load = load - delivered
        else:
            surplus = demand_threshold_kw - load
            if soc < 90.0:
                charged, soc = charge_kwh(battery, surplus, 1.0, soc)
                total_charged += charged
                net_load = load + charged
            else:
                net_load = load
        effective_peak = max(effective_peak, net_load)
        net_grid += net_load

    shaved = round(original_peak - effective_peak, 4)
    cost_saving = round(shaved * demand_charge_per_kw, 2)
    cycles = round(total_discharged / battery.capacity_kwh, 4)

    return DispatchResult(
        final_soc_pct=round(soc, 2),
        energy_charged_kwh=round(total_charged, 4),
        energy_discharged_kwh=round(total_discharged, 4),
        net_grid_kwh=round(net_grid, 4),
        peak_shaved_kw=shaved,
        cycle_count=cycles,
        cost_saving=cost_saving,
    )
