"""Battery energy storage system (BESS) modelling.

Simulates charge/discharge against a load and tariff to size a battery and
value peak shaving. Accounts for round-trip efficiency, depth-of-discharge
limits, and per-cycle degradation — the three things that separate nameplate
capacity from usable capacity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_ROUND_TRIP_EFFICIENCY: float = 0.90
"""Energy returned per unit stored, covering both conversion directions."""

DEFAULT_MAX_DEPTH_OF_DISCHARGE: float = 0.80
"""Fraction of nameplate capacity that may be cycled without harming life."""

DEFAULT_DEGRADATION_PER_CYCLE: float = 0.0002
"""Fractional capacity loss per full equivalent cycle."""


@dataclass
class BatterySpec:
    """Physical limits of a storage system.

    Args:
        capacity_kwh: Nameplate energy capacity.
        max_charge_kw: Maximum charge power.
        max_discharge_kw: Maximum discharge power.
        round_trip_efficiency: Energy returned per unit stored, in (0, 1].
        max_depth_of_discharge: Usable fraction of capacity, in (0, 1].
    """

    capacity_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    round_trip_efficiency: float = DEFAULT_ROUND_TRIP_EFFICIENCY
    max_depth_of_discharge: float = DEFAULT_MAX_DEPTH_OF_DISCHARGE

    def __post_init__(self) -> None:
        """Validate the specification.

        Raises:
            ValueError: If any capacity or power limit is not positive, or
                either fractional term falls outside (0, 1].
        """
        if self.capacity_kwh <= 0:
            raise ValueError(f"capacity_kwh must be positive, got {self.capacity_kwh}")
        if self.max_charge_kw <= 0 or self.max_discharge_kw <= 0:
            raise ValueError(
                f"power limits must be positive, got charge={self.max_charge_kw} discharge={self.max_discharge_kw}"
            )
        if not 0.0 < self.round_trip_efficiency <= 1.0:
            raise ValueError(f"round_trip_efficiency must be in (0, 1], got {self.round_trip_efficiency}")
        if not 0.0 < self.max_depth_of_discharge <= 1.0:
            raise ValueError(f"max_depth_of_discharge must be in (0, 1], got {self.max_depth_of_discharge}")

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
    """Discharge to hold load under a target, recharging when there is headroom.

    The battery discharges whenever load exceeds *target_peak_kw* and charges
    whenever load sits below it, subject to power limits, usable capacity,
    and round-trip efficiency. Charging losses are applied on the way in, so
    storing 1 kWh draws ``1 / efficiency`` kWh from the grid.

    Args:
        hourly_load_kw: Site demand per hour in kW.
        spec: Battery limits.
        target_peak_kw: Demand ceiling to defend.

    Returns:
        A populated :class:`DispatchResult`. When the battery cannot fully
        defend the target, ``peak_after_kw`` reports the peak actually achieved.

    Raises:
        ValueError: If *target_peak_kw* is negative or any load is negative.
    """
    if target_peak_kw < 0:
        raise ValueError(f"target_peak_kw must be non-negative, got {target_peak_kw}")
    for hour, load in enumerate(hourly_load_kw):
        if load < 0:
            raise ValueError(f"load must be non-negative, got {load} at hour {hour}")

    if not hourly_load_kw:
        return DispatchResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [])

    usable = spec.usable_kwh
    stored = usable  # Assume the battery starts charged for the event.
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
            # Charging is lossy: drawing `d` from the grid stores `d * efficiency`.
            drawn = min(headroom, room_in_pack / spec.round_trip_efficiency) if room_in_pack > 0 else 0.0
            stored += drawn * spec.round_trip_efficiency
            charged_total += drawn
            grid.append(round(load + drawn, 4))

    peak_before = round(max(hourly_load_kw), 4)
    peak_after = round(max(grid), 4)
    reduction = round(peak_before - peak_after, 4)
    cycles = round(discharged_total / usable, 4) if usable > 0 else 0.0

    if peak_after > target_peak_kw:
        logger.warning("Battery could not defend %.2f kW target; peak reached %.2f kW", target_peak_kw, peak_after)
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
    """Return the usable energy needed to hold load under a target.

    Sizes for the largest single excursion above the target, which is what
    the battery must ride through without recharging.

    Args:
        hourly_load_kw: Site demand per hour in kW.
        target_peak_kw: Demand ceiling to defend.

    Returns:
        Required usable capacity in kWh rounded to 4 decimal places.

    Raises:
        ValueError: If *target_peak_kw* is negative.
    """
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
    """Value a peak reduction against a monthly demand charge.

    Args:
        result: Outcome of a :func:`peak_shave` run.
        demand_charge_per_kw: Tariff charged per kW of billing peak.

    Returns:
        Saving rounded to 2 decimal places.

    Raises:
        ValueError: If *demand_charge_per_kw* is negative.
    """
    if demand_charge_per_kw < 0:
        raise ValueError(f"demand_charge_per_kw must be non-negative, got {demand_charge_per_kw}")
    return round(result.peak_reduction_kw * demand_charge_per_kw, 2)


def round_trip_losses_kwh(charged_kwh: float, efficiency: float) -> float:
    """Return the energy lost in a charge/discharge round trip.

    Args:
        charged_kwh: Energy drawn from the grid to charge.
        efficiency: Round-trip efficiency in (0, 1].

    Returns:
        Energy lost in kWh, rounded to 4 decimal places.

    Raises:
        ValueError: If *charged_kwh* is negative or *efficiency* is outside (0, 1].
    """
    if charged_kwh < 0:
        raise ValueError(f"charged_kwh must be non-negative, got {charged_kwh}")
    if not 0.0 < efficiency <= 1.0:
        raise ValueError(f"efficiency must be in (0, 1], got {efficiency}")
    return round(charged_kwh * (1.0 - efficiency), 4)


def break_even_cycles(capex: float, saving_per_cycle: float) -> float:
    """Return the cycles needed to recover a capital expenditure.

    Args:
        capex: Upfront cost in currency units.
        saving_per_cycle: Revenue or saving per equivalent full cycle.

    Returns:
        Cycles to break even, rounded to 2 decimal places. Returns
        ``float('inf')`` when *saving_per_cycle* is zero.

    Raises:
        ValueError: If either argument is negative.
    """
    if capex < 0:
        raise ValueError(f"capex must be non-negative, got {capex}")
    if saving_per_cycle < 0:
        raise ValueError(f"saving_per_cycle must be non-negative, got {saving_per_cycle}")
    if saving_per_cycle == 0:
        return float("inf")
    return round(capex / saving_per_cycle, 2)


__all__ = [
    "DEFAULT_DEGRADATION_PER_CYCLE",
    "DEFAULT_MAX_DEPTH_OF_DISCHARGE",
    "DEFAULT_ROUND_TRIP_EFFICIENCY",
    "BatterySpec",
    "DispatchResult",
    "break_even_cycles",
    "demand_charge_saving",
    "peak_shave",
    "required_capacity_kwh",
    "round_trip_losses_kwh",
]
