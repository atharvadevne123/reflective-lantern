"""Battery-based peak shaving optimisation for demand charge reduction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BatteryState:
    """Mutable battery state used during peak-shaving simulation."""

    capacity_kwh: float
    soc_kwh: float
    max_charge_rate_kw: float
    max_discharge_rate_kw: float
    round_trip_efficiency: float = 0.92

    def __post_init__(self) -> None:
        if self.capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be positive")
        if not (0 < self.round_trip_efficiency <= 1):
            raise ValueError("round_trip_efficiency must be in (0, 1]")
        if self.soc_kwh < 0 or self.soc_kwh > self.capacity_kwh:
            raise ValueError("soc_kwh must be in [0, capacity_kwh]")

    @property
    def soc_fraction(self) -> float:
        """State of charge as a fraction [0, 1]."""
        return self.soc_kwh / self.capacity_kwh

    def charge(self, power_kw: float, duration_h: float) -> float:
        """Charge the battery, respecting rate and capacity limits.

        Args:
            power_kw: Requested charge power in kW.
            duration_h: Duration of charge interval in hours.

        Returns:
            Actual energy charged in kWh (at AC side).
        """
        sqrt_eff = self.round_trip_efficiency**0.5
        max_energy = min(
            power_kw * duration_h,
            self.max_charge_rate_kw * duration_h,
            (self.capacity_kwh - self.soc_kwh) / sqrt_eff,
        )
        max_energy = max(0.0, max_energy)
        dc_energy = max_energy * sqrt_eff
        self.soc_kwh = min(self.capacity_kwh, self.soc_kwh + dc_energy)
        return round(max_energy, 6)

    def discharge(self, power_kw: float, duration_h: float) -> float:
        """Discharge the battery, respecting rate and SoC limits.

        Args:
            power_kw: Requested discharge power in kW.
            duration_h: Duration of discharge interval in hours.

        Returns:
            Actual energy delivered in kWh (at AC side).
        """
        sqrt_eff = self.round_trip_efficiency**0.5
        max_energy = min(
            power_kw * duration_h,
            self.max_discharge_rate_kw * duration_h,
            self.soc_kwh * sqrt_eff,
        )
        max_energy = max(0.0, max_energy)
        dc_energy = max_energy / sqrt_eff
        self.soc_kwh = max(0.0, self.soc_kwh - dc_energy)
        return round(max_energy, 6)


@dataclass
class PeakShavingResult:
    """Summary of a peak-shaving simulation run."""

    original_peak_kw: float
    shaved_peak_kw: float
    peak_reduction_kw: float
    peak_reduction_pct: float
    total_demand_kwh: float
    battery_cycles: float
    demand_charge_saved: float
    interval_loads_kw: list[float] = field(default_factory=list)
    interval_battery_kw: list[float] = field(default_factory=list)


def run_peak_shaving(
    load_profile_kw: list[float],
    battery: BatteryState,
    peak_threshold_kw: float,
    interval_hours: float = 0.25,
    demand_charge_per_kw: float = 15.0,
) -> PeakShavingResult:
    """Simulate peak shaving with a battery over a load profile.

    Args:
        load_profile_kw: Load profile as a list of power values in kW.
        battery: Mutable :class:`BatteryState` to use.
        peak_threshold_kw: Target peak demand limit in kW.
        interval_hours: Duration of each interval in hours (default 15 min = 0.25 h).
        demand_charge_per_kw: Demand charge rate in $/kW of monthly peak.

    Returns:
        Populated :class:`PeakShavingResult`.

    Raises:
        ValueError: If load_profile is empty or peak_threshold is non-positive.
    """
    if not load_profile_kw:
        raise ValueError("load_profile_kw must not be empty")
    if peak_threshold_kw <= 0:
        raise ValueError("peak_threshold_kw must be positive")

    net_loads: list[float] = []
    battery_powers: list[float] = []
    original_peak = max(load_profile_kw)

    for load in load_profile_kw:
        excess = load - peak_threshold_kw
        if excess > 0:
            discharged = battery.discharge(excess, interval_hours)
            net = load - discharged / interval_hours
            battery_powers.append(-discharged / interval_hours)
        else:
            headroom = peak_threshold_kw - load
            charged = battery.charge(headroom, interval_hours)
            net = load + charged / interval_hours
            battery_powers.append(charged / interval_hours)
        net_loads.append(round(net, 4))

    shaved_peak = max(net_loads)
    reduction = original_peak - shaved_peak
    reduction_pct = 100.0 * reduction / original_peak if original_peak > 0 else 0.0
    total_kwh = sum(load_profile_kw) * interval_hours
    avg_discharge = sum(abs(p) for p in battery_powers if p < 0) * interval_hours
    cycles = avg_discharge / battery.capacity_kwh if battery.capacity_kwh > 0 else 0.0
    demand_saved = demand_charge_per_kw * reduction

    return PeakShavingResult(
        original_peak_kw=round(original_peak, 4),
        shaved_peak_kw=round(shaved_peak, 4),
        peak_reduction_kw=round(reduction, 4),
        peak_reduction_pct=round(reduction_pct, 3),
        total_demand_kwh=round(total_kwh, 4),
        battery_cycles=round(cycles, 4),
        demand_charge_saved=round(demand_saved, 2),
        interval_loads_kw=net_loads,
        interval_battery_kw=[round(p, 4) for p in battery_powers],
    )


def optimal_threshold(
    load_profile_kw: list[float],
    battery_capacity_kwh: float,
    battery_power_kw: float,
    percentile: float = 85.0,
) -> float:
    """Heuristic peak threshold based on a load percentile.

    Args:
        load_profile_kw: Historical load profile in kW.
        battery_capacity_kwh: Available battery capacity in kWh.
        battery_power_kw: Battery rated power in kW.
        percentile: Percentile of load to use as threshold (default 85th).

    Returns:
        Suggested peak threshold in kW.

    Raises:
        ValueError: If load_profile is empty or percentile outside [0, 100].
    """
    if not load_profile_kw:
        raise ValueError("load_profile_kw must not be empty")
    if not (0 <= percentile <= 100):
        raise ValueError("percentile must be in [0, 100]")
    sorted_loads = sorted(load_profile_kw)
    idx = int(len(sorted_loads) * percentile / 100.0)
    idx = min(idx, len(sorted_loads) - 1)
    return round(sorted_loads[idx], 4)


def demand_charge_reduction(
    original_peak_kw: float,
    shaved_peak_kw: float,
    demand_rate_per_kw: float,
) -> float:
    """Calculate monthly demand charge savings.

    Args:
        original_peak_kw: Peak demand before shaving in kW.
        shaved_peak_kw: Peak demand after shaving in kW.
        demand_rate_per_kw: Demand charge rate in $/kW.

    Returns:
        Monthly savings in dollars.

    Raises:
        ValueError: If shaved peak exceeds original peak.
    """
    if shaved_peak_kw > original_peak_kw:
        raise ValueError("shaved_peak_kw cannot exceed original_peak_kw")
    return round((original_peak_kw - shaved_peak_kw) * demand_rate_per_kw, 2)


def battery_cycles_per_year(
    daily_charge_kwh: float,
    battery_capacity_kwh: float,
) -> float:
    """Return the equivalent full cycles per year based on daily charge throughput.

    Args:
        daily_charge_kwh: Average daily energy throughput in kWh.
        battery_capacity_kwh: Usable battery capacity in kWh (must be positive).

    Returns:
        Equivalent full cycles per year.

    Raises:
        ValueError: If battery_capacity_kwh ≤ 0 or daily_charge_kwh < 0.
    """
    if battery_capacity_kwh <= 0:
        raise ValueError("battery_capacity_kwh must be positive")
    if daily_charge_kwh < 0:
        raise ValueError("daily_charge_kwh must be non-negative")
    return (daily_charge_kwh * 365.0) / battery_capacity_kwh


def revenue_from_peak_shaving(
    demand_reduction_kw: float,
    demand_charge_per_kw: float,
    months: int = 12,
) -> float:
    """Return annual revenue from demand charge reduction.

    Args:
        demand_reduction_kw: Reduction in peak demand in kW (non-negative).
        demand_charge_per_kw: Demand charge tariff in currency/kW/month.
        months: Number of months of benefit (default 12).

    Returns:
        Annual revenue from avoided demand charges in currency units.

    Raises:
        ValueError: If demand_reduction_kw < 0 or demand_charge_per_kw < 0.
    """
    if demand_reduction_kw < 0:
        raise ValueError("demand_reduction_kw must be non-negative")
    if demand_charge_per_kw < 0:
        raise ValueError("demand_charge_per_kw must be non-negative")
    return demand_reduction_kw * demand_charge_per_kw * months


def shaving_effectiveness(original_peak_kw: float, shaved_peak_kw: float) -> float:
    """Return peak shaving effectiveness as a fraction in [0, 1].

    Args:
        original_peak_kw: Original peak demand before shaving in kW.
        shaved_peak_kw: Peak demand after shaving in kW.

    Returns:
        Fraction of peak demand that was shaved.

    Raises:
        ValueError: If original_peak_kw ≤ 0 or shaved_peak_kw > original_peak_kw.
    """
    if original_peak_kw <= 0:
        raise ValueError("original_peak_kw must be positive")
    if shaved_peak_kw > original_peak_kw:
        raise ValueError("shaved_peak_kw cannot exceed original_peak_kw")
    return (original_peak_kw - shaved_peak_kw) / original_peak_kw
