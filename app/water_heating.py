"""Domestic hot water heating energy estimation and heat-pump comparison."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

WATER_SPECIFIC_HEAT_KJ_PER_KG_K: float = 4.186
"""Specific heat capacity of water in kJ/(kg·K)."""

WATER_DENSITY_KG_PER_LITRE: float = 1.0
"""Density of water in kg/L (at ~20°C, simplification)."""

KJ_PER_KWH: float = 3600.0
"""Kilojoules per kilowatt-hour."""


@dataclass
class DHWResult:
    """Domestic hot water energy and cost estimate."""

    daily_litres: float
    daily_kwh_useful: float
    daily_kwh_consumed: float
    daily_cost: float
    annual_kwh_consumed: float
    annual_cost: float
    technology: str
    cop_or_efficiency: float


def energy_to_heat_water(
    litres: float,
    cold_temp_c: float,
    hot_temp_c: float,
) -> float:
    """Calculate useful thermal energy to heat a volume of water.

    Args:
        litres: Volume of water in litres.
        cold_temp_c: Inlet (cold) temperature in °C.
        hot_temp_c: Target (hot) temperature in °C.

    Returns:
        Useful thermal energy in kWh.

    Raises:
        ValueError: If litres is negative or hot_temp <= cold_temp.
    """
    if litres < 0:
        raise ValueError("litres must be non-negative")
    if hot_temp_c <= cold_temp_c:
        raise ValueError("hot_temp_c must be greater than cold_temp_c")
    mass_kg = litres * WATER_DENSITY_KG_PER_LITRE
    delta_t = hot_temp_c - cold_temp_c
    kj = mass_kg * WATER_SPECIFIC_HEAT_KJ_PER_KG_K * delta_t
    return round(kj / KJ_PER_KWH, 6)


def boiler_consumption(
    useful_kwh: float,
    boiler_efficiency: float,
) -> float:
    """Convert useful heat demand to gas boiler fuel consumption.

    Args:
        useful_kwh: Useful heat demand in kWh.
        boiler_efficiency: Seasonal boiler efficiency in (0, 1].

    Returns:
        Fuel consumed in kWh (gas).

    Raises:
        ValueError: If useful_kwh is negative or efficiency is outside (0, 1].
    """
    if useful_kwh < 0:
        raise ValueError("useful_kwh must be non-negative")
    if not (0 < boiler_efficiency <= 1):
        raise ValueError("boiler_efficiency must be in (0, 1]")
    return round(useful_kwh / boiler_efficiency, 6)


def heat_pump_consumption(
    useful_kwh: float,
    cop: float,
) -> float:
    """Convert useful heat demand to heat-pump electricity consumption.

    Args:
        useful_kwh: Useful heat demand in kWh.
        cop: Coefficient of performance (>= 1.0).

    Returns:
        Electricity consumed in kWh.

    Raises:
        ValueError: If useful_kwh is negative or cop < 1.
    """
    if useful_kwh < 0:
        raise ValueError("useful_kwh must be non-negative")
    if cop < 1.0:
        raise ValueError("cop must be >= 1.0")
    return round(useful_kwh / cop, 6)


def estimate_dhw(
    daily_litres: float,
    cold_temp_c: float = 10.0,
    hot_temp_c: float = 60.0,
    technology: str = "boiler",
    cop_or_efficiency: float = 0.90,
    energy_rate: float = 0.28,
) -> DHWResult:
    """Full domestic hot water estimate.

    Args:
        daily_litres: Hot water demand per day in litres.
        cold_temp_c: Mains water temperature in °C.
        hot_temp_c: Target delivery temperature in °C.
        technology: Either ``'boiler'`` or ``'heat_pump'``.
        cop_or_efficiency: Boiler efficiency (0-1) or heat-pump COP (>= 1).
        energy_rate: Energy cost in $/kWh.

    Returns:
        Populated :class:`DHWResult`.

    Raises:
        ValueError: If technology is not ``'boiler'`` or ``'heat_pump'``.
    """
    if technology not in ("boiler", "heat_pump"):
        raise ValueError("technology must be 'boiler' or 'heat_pump'")
    useful = energy_to_heat_water(daily_litres, cold_temp_c, hot_temp_c)
    if technology == "boiler":
        consumed = boiler_consumption(useful, cop_or_efficiency)
    else:
        consumed = heat_pump_consumption(useful, cop_or_efficiency)
    daily_cost = round(consumed * energy_rate, 4)
    return DHWResult(
        daily_litres=daily_litres,
        daily_kwh_useful=useful,
        daily_kwh_consumed=consumed,
        daily_cost=daily_cost,
        annual_kwh_consumed=round(consumed * 365, 4),
        annual_cost=round(daily_cost * 365, 4),
        technology=technology,
        cop_or_efficiency=cop_or_efficiency,
    )


def compare_technologies(
    daily_litres: float,
    cold_temp_c: float = 10.0,
    hot_temp_c: float = 60.0,
    boiler_efficiency: float = 0.90,
    boiler_gas_rate: float = 0.07,
    heat_pump_cop: float = 3.0,
    electricity_rate: float = 0.28,
) -> dict[str, DHWResult]:
    """Compare boiler vs heat-pump for domestic hot water.

    Args:
        daily_litres: Hot water demand per day in litres.
        cold_temp_c: Inlet temperature in °C.
        hot_temp_c: Target temperature in °C.
        boiler_efficiency: Gas boiler seasonal efficiency.
        boiler_gas_rate: Gas tariff in $/kWh.
        heat_pump_cop: ASHP COP for hot-water heating.
        electricity_rate: Electricity tariff in $/kWh.

    Returns:
        Dict with keys ``'boiler'`` and ``'heat_pump'``, each a :class:`DHWResult`.
    """
    return {
        "boiler": estimate_dhw(
            daily_litres, cold_temp_c, hot_temp_c, "boiler", boiler_efficiency, boiler_gas_rate
        ),
        "heat_pump": estimate_dhw(
            daily_litres, cold_temp_c, hot_temp_c, "heat_pump", heat_pump_cop, electricity_rate
        ),
    }
