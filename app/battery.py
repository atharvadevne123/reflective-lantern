"""Battery energy storage system (BESS) modelling.

Simulates charge/discharge against a load and tariff to size a battery and
value peak shaving. Accounts for round-trip efficiency, depth-of-discharge
limits, and per-cycle degradation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_ROUND_TRIP_EFFICIENCY: float = 0.90
DEFAULT_MAX_DEPTH_OF_DISCHARGE: float = 0.80
DEFAULT_DEGRADATION_PER_CYCLE: float = 0.0002


@dataclass
class BatterySpec:
    """Physical limits of a storage system."""

    capacity_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    round_trip_efficiency: float = DEFAULT_ROUND_TRIP_EFFICIENCY
    max_depth_of_discharge: float = DEFAULT_MAX_DEPTH_OF_DISCHARGE

    def __post_init__(self) -> None:
        if self.capacity_kwh <= 0:
            raise ValueError(f"capacity_kwh must be positive, got {self.capacity_kwh}")
        if self.max_charge_kw <= 0 or self.max_discharge_kw <= 0:
            raise ValueError(
                f"power limits must be positive, got charge={self.max_charge_kw} discharge={self.max_discharge_kw}"
            )
        if not 0.0 < self.round_trip_efficiency <= 1.0:
            raise ValueError(
                f"round_trip_efficiency must be in (0, 1], got {self.round_trip_efficiency}"
            )
        if not 0.0 < self.max_depth_of_discharge <= 1.0:
            raise ValueError(
                f"max_depth_of_discharge must be in (0, 1], got {self.max_depth_of_discharge}"
            )

    @property
    def usable_kwh(self) -> float:
        """Return the capacity that may actually be cycled."""
        return round(self.capacity_kwh * self.max_depth_of_discharge, 4)


@dataclass
class DispatchResult:
    """Outcome of simulating a battery against an hourly load."""

    peak_before_kw: float
    peak_after_kw: float
    peak_reduction_kw: float
    peak_reduction_pct: float
    energy_discharged_kwh: float
    energy_charged_kwh: float
    equivalent_cycles: float
    capacity_lost_pct: float
    grid_hourly_kw: list[float]


def peak_shave(
    hourly_load_kw: list[float],
    spec: BatterySpec,
    target_peak_kw: float,
) -> DispatchResult:
    """Discharge to hold load under a target, recharging when there is headroom."""
    if target_peak_kw < 0:
        raise ValueError(f"target_peak_kw must be non-negative, got {target_peak_kw}")
    for hour, load in enumerate(hourly_load_kw):
        if load < 0:
            raise ValueError(f"load must be non-negative, got {load} at hour {hour}")

    if not hourly_load_kw:
        return DispatchResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [])

    usable = spec.usable_kwh
    stored = usable
    discharged_total = charged_total = 0.0
    grid: list[float] = []

    for load in hourly_load_kw:
        if load > target_peak_kw:
            wanted = min(load - target_peak_kw, spec.max_discharge_kw, stored)
            stored -= wanted
            discharged_total += wanted
            grid.append(round(load - wanted, 4))
        else:
            headroom = min(target_peak_kw - load, spec.max_charge_kw)
            room_in_pack = usable - stored
            drawn = (
                min(headroom, room_in_pack / spec.round_trip_efficiency)
                if room_in_pack > 0
                else 0.0
            )
            stored += drawn * spec.round_trip_efficiency
            charged_total += drawn
            grid.append(round(load + drawn, 4))

    peak_before = round(max(hourly_load_kw), 4)
    peak_after = round(max(grid), 4)
    reduction = round(peak_before - peak_after, 4)
    cycles = round(discharged_total / usable, 4) if usable > 0 else 0.0

    if peak_after > target_peak_kw:
        logger.warning(
            "Battery could not defend %.2f kW target; peak reached %.2f kW",
            target_peak_kw,
            peak_after,
        )
    return DispatchResult(
        peak_before_kw=peak_before,
        peak_after_kw=peak_after,
        peak_reduction_kw=reduction,
        peak_reduction_pct=round(100.0 * reduction / peak_before, 2) if peak_before > 0 else 0.0,
        energy_discharged_kwh=round(discharged_total, 4),
        energy_charged_kwh=round(charged_total, 4),
        equivalent_cycles=cycles,
        capacity_lost_pct=round(100.0 * cycles * DEFAULT_DEGRADATION_PER_CYCLE, 6),
        grid_hourly_kw=grid,
    )


def required_capacity_kwh(hourly_load_kw: list[float], target_peak_kw: float) -> float:
    """Return the usable energy needed to hold load under a target."""
    if target_peak_kw < 0:
        raise ValueError(f"target_peak_kw must be non-negative, got {target_peak_kw}")

    largest = running = 0.0
    for load in hourly_load_kw:
        if load > target_peak_kw:
            running += load - target_peak_kw
            largest = max(largest, running)
        else:
            running = 0.0
    return round(largest, 4)


def demand_charge_saving(
    result: DispatchResult,
    demand_charge_per_kw: float,
) -> float:
    """Value a peak reduction against a monthly demand charge."""
    if demand_charge_per_kw < 0:
        raise ValueError(f"demand_charge_per_kw must be non-negative, got {demand_charge_per_kw}")
    return round(result.peak_reduction_kw * demand_charge_per_kw, 2)


def round_trip_losses_kwh(charged_kwh: float, efficiency: float) -> float:
    """Return the energy lost in a charge/discharge round trip."""
    if charged_kwh < 0:
        raise ValueError(f"charged_kwh must be non-negative, got {charged_kwh}")
    if not 0.0 < efficiency <= 1.0:
        raise ValueError(f"efficiency must be in (0, 1], got {efficiency}")
    return round(charged_kwh * (1.0 - efficiency), 4)


def break_even_cycles(capex: float, saving_per_cycle: float) -> float:
    """Return the cycles needed to recover a capital expenditure."""
    if capex < 0:
        raise ValueError(f"capex must be non-negative, got {capex}")
    if saving_per_cycle < 0:
        raise ValueError(f"saving_per_cycle must be non-negative, got {saving_per_cycle}")
    if saving_per_cycle == 0:
        return float("inf")
    return round(capex / saving_per_cycle, 2)


def usable_capacity_kwh(
    nameplate_kwh: float, max_dod: float = DEFAULT_MAX_DEPTH_OF_DISCHARGE
) -> float:
    """Return the usable energy capacity after applying depth-of-discharge."""
    if nameplate_kwh < 0:
        raise ValueError(f"nameplate_kwh must be non-negative, got {nameplate_kwh}")
    if not 0.0 < max_dod <= 1.0:
        raise ValueError(f"max_dod must be in (0, 1], got {max_dod}")
    return round(nameplate_kwh * max_dod, 4)


def cycles_to_degrade(
    initial_capacity_kwh: float,
    target_fraction: float = 0.8,
    degradation_per_cycle: float = DEFAULT_DEGRADATION_PER_CYCLE,
) -> int:
    """Estimate number of cycles until capacity drops to *target_fraction* of initial.

    Args:
        initial_capacity_kwh: Starting capacity in kWh.
        target_fraction: Fraction of initial capacity at which degraded.
        degradation_per_cycle: Fractional capacity lost per full cycle.

    Returns:
        Integer number of cycles until the target is reached.

    Raises:
        ValueError: If *target_fraction* not in (0, 1] or *degradation_per_cycle* not positive.
    """
    if not (0 < target_fraction <= 1.0):
        raise ValueError("target_fraction must be in (0, 1]")
    if degradation_per_cycle <= 0:
        raise ValueError("degradation_per_cycle must be positive")
    if target_fraction >= 1.0:
        return 0
    import math

    cycles = math.log(target_fraction) / math.log(1 - degradation_per_cycle)
    return max(0, int(cycles))


__all__ = [
    "DEFAULT_DEGRADATION_PER_CYCLE",
    "DEFAULT_MAX_DEPTH_OF_DISCHARGE",
    "DEFAULT_ROUND_TRIP_EFFICIENCY",
    "BatterySpec",
    "DispatchResult",
    "break_even_cycles",
    "cycles_to_degrade",
    "demand_charge_saving",
    "peak_shave",
    "required_capacity_kwh",
    "round_trip_losses_kwh",
    "usable_capacity_kwh",
]
