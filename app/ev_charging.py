"""Electric vehicle charging load modelling and smart charging optimisation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class ChargerLevel(StrEnum):
    """EV charger power level classifications."""

    LEVEL_1 = "Level 1"
    LEVEL_2 = "Level 2"
    DC_FAST = "DC Fast"


CHARGER_POWER_KW: dict[ChargerLevel, float] = {
    ChargerLevel.LEVEL_1: 1.4,
    ChargerLevel.LEVEL_2: 7.2,
    ChargerLevel.DC_FAST: 50.0,
}


@dataclass
class ChargingSession:
    """A single EV charging session."""

    battery_kwh: float = 0.0
    start_soc_pct: float = 0.0
    target_soc_pct: float = 100.0
    charger_level: ChargerLevel = ChargerLevel.LEVEL_2
    charge_efficiency: float = 0.90
    # Alias fields — mapped in __post_init__
    battery_capacity_kwh: float | None = None
    initial_soc: float | None = None
    target_soc: float | None = None

    def __post_init__(self) -> None:
        if self.battery_capacity_kwh is not None:
            self.battery_kwh = self.battery_capacity_kwh
        if self.initial_soc is not None:
            self.start_soc_pct = self.initial_soc * 100.0 if self.initial_soc <= 1.0 else self.initial_soc
        if self.target_soc is not None:
            self.target_soc_pct = self.target_soc * 100.0 if self.target_soc <= 1.0 else self.target_soc


@dataclass
class ChargingResult:
    """Result of an EV charging session calculation."""

    energy_added_kwh: float
    energy_drawn_kwh: float
    duration_hours: float
    cost: float
    charger_level: str
    charging_rate_kw: float

    @property
    def energy_kwh(self) -> float:
        """Alias for energy_drawn_kwh."""
        return self.energy_drawn_kwh

    def to_dict(self) -> dict[str, float | str]:
        """Return charging result as a plain dict."""
        return {
            "energy_added_kwh": self.energy_added_kwh,
            "energy_drawn_kwh": self.energy_drawn_kwh,
            "duration_hours": self.duration_hours,
            "cost": self.cost,
            "charger_level": self.charger_level,
            "charging_rate_kw": self.charging_rate_kw,
        }


def energy_to_charge(
    start_or_battery: float,
    target_or_start: float,
    battery_or_target: float,
    charge_efficiency: float = 0.90,
) -> float:
    """Compute energy drawn from the grid to charge an EV battery.

    Accepts two calling conventions:
      - energy_to_charge(start_soc_pct, target_soc_pct, battery_kwh[, efficiency])
      - energy_to_charge(battery_kwh, start_soc_decimal, target_soc_decimal)

    Returns:
        Grid energy consumed in kWh.

    Raises:
        ValueError: If SOC values are out of range, start >= target, or efficiency invalid.
    """
    # Detect convention: if second arg <= 1.1 and third arg <= 1.0, new decimal convention
    _new_conv = target_or_start <= 1.1 and battery_or_target <= 1.0 and start_or_battery > 1.0
    if _new_conv:
        # New convention: (battery_kwh, start_decimal, target_decimal)
        battery_kwh = start_or_battery
        start_soc_pct = target_or_start * 100.0
        target_soc_pct = battery_or_target * 100.0
        if target_or_start > 1.0:
            raise ValueError("SOC values must be in [0, 100]")
    else:
        # Legacy convention: (start_soc_pct, target_soc_pct, battery_kwh)
        start_soc_pct = start_or_battery
        target_soc_pct = target_or_start
        battery_kwh = battery_or_target
    if not (0 <= start_soc_pct <= 100) or not (0 <= target_soc_pct <= 100):
        raise ValueError("SOC values must be in [0, 100]")
    if start_soc_pct > target_soc_pct:
        raise ValueError("target_soc_pct must exceed start_soc_pct")
    if start_soc_pct == target_soc_pct:
        if not _new_conv:
            raise ValueError("target_soc_pct must exceed start_soc_pct")
        return 0.0
    if not (0 < charge_efficiency <= 1):
        raise ValueError("charge_efficiency must be in (0, 1]")
    energy_added = battery_kwh * (target_soc_pct - start_soc_pct) / 100.0
    return round(energy_added / charge_efficiency, 4)


def charging_duration_hours(
    energy_kwh: float,
    charger_level: ChargerLevel | float,
) -> float:
    """Estimate charging duration for a given energy demand.

    Args:
        energy_kwh: Total grid energy to deliver in kWh.
        charger_level: Charger power level (ChargerLevel enum or kW float directly).

    Returns:
        Duration in hours.

    Raises:
        ValueError: If energy_kwh is negative.
    """
    if energy_kwh < 0:
        raise ValueError("energy_kwh must be non-negative")
    if energy_kwh == 0.0:
        return 0.0
    if isinstance(charger_level, (int, float)):
        rate = float(charger_level)
    else:
        rate = CHARGER_POWER_KW[charger_level]
    return round(energy_kwh / rate, 4)


def simulate_session(
    session: ChargingSession,
    energy_rate: float = 0.28,
) -> ChargingResult:
    """Simulate a full EV charging session.

    Args:
        session: :class:`ChargingSession` parameters.
        energy_rate: Electricity cost in $/kWh.

    Returns:
        Populated :class:`ChargingResult`.
    """
    grid_kwh = energy_to_charge(
        session.start_soc_pct,
        session.target_soc_pct,
        session.battery_kwh,
        session.charge_efficiency,
    )
    energy_added = grid_kwh * session.charge_efficiency
    duration = charging_duration_hours(grid_kwh, session.charger_level)
    rate = CHARGER_POWER_KW[session.charger_level]
    return ChargingResult(
        energy_added_kwh=round(energy_added, 4),
        energy_drawn_kwh=grid_kwh,
        duration_hours=duration,
        cost=round(grid_kwh * energy_rate, 4),
        charger_level=session.charger_level.value,
        charging_rate_kw=rate,
    )


def fleet_daily_demand_kw(
    num_vehicles: int,
    avg_daily_miles: float,
    efficiency_miles_per_kwh: float,
    charge_efficiency: float = 0.90,
    hours_available: float = 8.0,
) -> float:
    """Estimate average grid power demand to charge a fleet overnight.

    Args:
        num_vehicles: Number of EVs in the fleet.
        avg_daily_miles: Average miles driven per vehicle per day.
        efficiency_miles_per_kwh: Vehicle energy efficiency (miles/kWh).
        charge_efficiency: Charging efficiency (default 90%).
        hours_available: Window available for charging in hours (default 8h).

    Returns:
        Average grid power demand in kW.

    Raises:
        ValueError: If any argument is non-positive.
    """
    if any(
        v <= 0 for v in (num_vehicles, avg_daily_miles, efficiency_miles_per_kwh, hours_available)
    ):
        raise ValueError("All arguments must be positive")
    if not (0 < charge_efficiency <= 1):
        raise ValueError("charge_efficiency must be in (0, 1]")
    daily_kwh_per_vehicle = avg_daily_miles / efficiency_miles_per_kwh
    grid_kwh_per_vehicle = daily_kwh_per_vehicle / charge_efficiency
    total_grid_kwh = grid_kwh_per_vehicle * num_vehicles
    return round(total_grid_kwh / hours_available, 4)


def smart_charge_schedule(
    total_kwh: float,
    hours_available: int,
    tou_rates: list[float],
) -> list[float]:
    """Greedily schedule charging to minimise cost using TOU rates.

    Allocates charging to the cheapest available hours first.

    Args:
        total_kwh: Total energy required.
        hours_available: Number of hours in the charging window.
        tou_rates: Per-hour energy rates ($/kWh), length must equal hours_available.

    Returns:
        List of kWh allocated to each hour.

    Raises:
        ValueError: If lengths mismatch or total_kwh is negative.
    """
    if total_kwh < 0:
        raise ValueError("total_kwh must be non-negative")
    if len(tou_rates) != hours_available:
        raise ValueError("tou_rates length must equal hours_available")
    schedule = [0.0] * hours_available
    sorted_hours = sorted(range(hours_available), key=lambda i: tou_rates[i])
    remaining = total_kwh
    per_hour = total_kwh / hours_available if hours_available > 0 else 0.0
    for idx, h in enumerate(sorted_hours):
        if remaining <= 0:
            break
        if idx == len(sorted_hours) - 1:
            schedule[h] = round(remaining, 6)
        else:
            schedule[h] = round(min(per_hour, remaining), 6)
        remaining = round(remaining - schedule[h], 8)
    return schedule


def charging_cost(
    energy_drawn_kwh: float,
    energy_rate: float,
    connection_fee: float = 0.0,
) -> float:
    """Return total cost of a charging session.

    Args:
        energy_drawn_kwh: Energy drawn from the charger in kWh (non-negative).
        energy_rate: Electricity tariff in currency/kWh (non-negative).
        connection_fee: Fixed session connection fee in currency (default 0).

    Returns:
        Total cost in currency units.

    Raises:
        ValueError: If energy_drawn_kwh < 0 or energy_rate < 0.
    """
    if energy_drawn_kwh < 0:
        raise ValueError("energy_drawn_kwh must be non-negative")
    if energy_rate < 0:
        raise ValueError("energy_rate must be non-negative")
    return energy_drawn_kwh * energy_rate + connection_fee


def v2g_discharge_kw(
    battery_soc_pct: float,
    battery_kwh: float,
    min_soc_pct: float = 20.0,
    charger_power_kw: float = 7.4,
) -> float:
    """Return available vehicle-to-grid (V2G) discharge power in kW.

    Discharging is limited by the available energy above the minimum SoC
    and by the charger's rated power.

    Args:
        battery_soc_pct: Current battery state of charge in percent (0–100).
        battery_kwh: Battery usable capacity in kWh (must be positive).
        min_soc_pct: Minimum SoC to retain for driving range (default 20%).
        charger_power_kw: Maximum V2G charger power in kW (default 7.4).

    Returns:
        Available V2G power in kW (0 if SoC ≤ min_soc_pct).

    Raises:
        ValueError: If battery_kwh ≤ 0 or SoC values out of range.
    """
    if battery_kwh <= 0:
        raise ValueError("battery_kwh must be positive")
    if not 0 <= battery_soc_pct <= 100:
        raise ValueError("battery_soc_pct must be in [0, 100]")
    if not 0 <= min_soc_pct < 100:
        raise ValueError("min_soc_pct must be in [0, 100)")
    available_kwh = max(0.0, (battery_soc_pct - min_soc_pct) / 100.0 * battery_kwh)
    if available_kwh <= 0:
        return 0.0
    return min(charger_power_kw, available_kwh)


def co2_saved_vs_ice(
    energy_added_kwh: float,
    grid_intensity_kg_per_kwh: float = 0.233,
    ice_emission_kg_per_km: float = 0.120,
    kwh_per_km: float = 0.20,
) -> float:
    """Return CO2 saved by charging an EV compared to an equivalent ICE vehicle.

    Args:
        energy_added_kwh: Usable energy added to the battery in kWh.
        grid_intensity_kg_per_kwh: Grid carbon intensity in kg CO2e/kWh.
        ice_emission_kg_per_km: ICE vehicle emissions in kg CO2e/km.
        kwh_per_km: EV energy consumption in kWh/km.

    Returns:
        Net CO2 savings in kg CO2e (positive = EV emits less than ICE).

    Raises:
        ValueError: If energy_added_kwh < 0.
    """
    if energy_added_kwh < 0:
        raise ValueError("energy_added_kwh must be non-negative")
    ev_emissions = energy_added_kwh * grid_intensity_kg_per_kwh
    km_driven = energy_added_kwh / kwh_per_km
    ice_emissions = km_driven * ice_emission_kg_per_km
    return ice_emissions - ev_emissions


__all__ = [
    "ChargerLevel",
    "ChargingResult",
    "ChargingSession",
    "charging_cost",
    "charging_duration_hours",
    "co2_saved_vs_ice",
    "energy_to_charge",
    "fleet_daily_demand_kw",
    "simulate_session",
    "smart_charge_schedule",
    "v2g_discharge_kw",
]
