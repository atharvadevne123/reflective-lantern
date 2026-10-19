"""HVAC energy consumption estimation using degree-day methods."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_BASE_TEMPERATURE: float = 18.0
"""Standard base temperature for UK degree-day calculations (°C)."""

DEFAULT_BOILER_EFFICIENCY: float = 0.85
"""Typical condensing boiler seasonal efficiency."""

DEFAULT_HEAT_PUMP_COP: float = 3.0
"""Representative air-source heat pump coefficient of performance."""


@dataclass
class HVACEstimate:
    """HVAC energy and cost estimate."""

    heating_kwh: float
    cooling_kwh: float
    total_kwh: float
    estimated_cost: float
    hdd: float
    cdd: float


def heating_degree_days(
    daily_temps_c: list[float],
    base_temp: float = DEFAULT_BASE_TEMPERATURE,
) -> float:
    """Compute Heating Degree Days (HDD) for a sequence of daily mean temperatures.

    Args:
        daily_temps_c: Daily mean temperatures in °C.
        base_temp: Base temperature in °C.

    Returns:
        Sum of HDD over the period (non-negative).

    Raises:
        ValueError: If daily_temps_c is empty.
    """
    if not daily_temps_c:
        raise ValueError("daily_temps_c must not be empty")
    return round(sum(max(0.0, base_temp - t) for t in daily_temps_c), 4)


def cooling_degree_days(
    daily_temps_c: list[float],
    base_temp: float = DEFAULT_BASE_TEMPERATURE,
) -> float:
    """Compute Cooling Degree Days (CDD) for a sequence of daily mean temperatures.

    Args:
        daily_temps_c: Daily mean temperatures in °C.
        base_temp: Base temperature in °C.

    Returns:
        Sum of CDD over the period (non-negative).

    Raises:
        ValueError: If daily_temps_c is empty.
    """
    if not daily_temps_c:
        raise ValueError("daily_temps_c must not be empty")
    return round(sum(max(0.0, t - base_temp) for t in daily_temps_c), 4)


def heating_kwh_from_hdd(
    hdd: float,
    heat_loss_coefficient_kw_per_k: float,
    hours_per_day: float = 24.0,
    efficiency: float = DEFAULT_BOILER_EFFICIENCY,
) -> float:
    """Estimate heating energy from HDD and building thermal characteristics.

    Args:
        hdd: Heating degree days.
        heat_loss_coefficient_kw_per_k: Building fabric + ventilation losses in kW/K.
        hours_per_day: Hours per day the heating system operates (default 24).
        efficiency: Heating system seasonal efficiency (e.g. 0.85 for boiler).

    Returns:
        Estimated heating energy in kWh.

    Raises:
        ValueError: If any argument is negative, or efficiency is outside (0, 1].
    """
    if hdd < 0 or heat_loss_coefficient_kw_per_k < 0:
        raise ValueError("hdd and heat_loss_coefficient must be non-negative")
    if not (0 < efficiency <= 1):
        raise ValueError("efficiency must be in (0, 1]")
    useful_heat = hdd * heat_loss_coefficient_kw_per_k * hours_per_day
    return round(useful_heat / efficiency, 4)


def cooling_kwh_from_cdd(
    cdd: float,
    sensible_heat_gain_kw_per_k: float,
    cop: float = DEFAULT_HEAT_PUMP_COP,
    hours_per_day: float = 24.0,
) -> float:
    """Estimate cooling energy from CDD and building thermal gain characteristics.

    Args:
        cdd: Cooling degree days.
        sensible_heat_gain_kw_per_k: Sensible heat gain rate in kW/K.
        cop: Cooling system COP (default 3.0).
        hours_per_day: Hours per day cooling operates (default 24).

    Returns:
        Estimated cooling electricity in kWh.

    Raises:
        ValueError: If cdd or sensible_heat_gain_kw_per_k is negative, or cop <= 0.
    """
    if cdd < 0 or sensible_heat_gain_kw_per_k < 0:
        raise ValueError("cdd and sensible_heat_gain must be non-negative")
    if cop <= 0:
        raise ValueError("cop must be positive")
    heat_removed = cdd * sensible_heat_gain_kw_per_k * hours_per_day
    return round(heat_removed / cop, 4)


def estimate_hvac(
    daily_temps_c: list[float],
    heat_loss_kw_per_k: float,
    sensible_heat_gain_kw_per_k: float,
    energy_rate: float = 0.28,
    boiler_efficiency: float = DEFAULT_BOILER_EFFICIENCY,
    cooling_cop: float = DEFAULT_HEAT_PUMP_COP,
) -> HVACEstimate:
    """Full HVAC energy and cost estimate for a temperature series.

    Args:
        daily_temps_c: Daily mean temperatures in °C.
        heat_loss_kw_per_k: Fabric and ventilation heat loss coefficient in kW/K.
        sensible_heat_gain_kw_per_k: Sensible heat gain per degree above base in kW/K.
        energy_rate: Energy cost in $/kWh.
        boiler_efficiency: Heating system efficiency.
        cooling_cop: Cooling COP.

    Returns:
        Populated :class:`HVACEstimate`.
    """
    hdd = heating_degree_days(daily_temps_c)
    cdd = cooling_degree_days(daily_temps_c)
    h_kwh = heating_kwh_from_hdd(hdd, heat_loss_kw_per_k, efficiency=boiler_efficiency)
    c_kwh = cooling_kwh_from_cdd(cdd, sensible_heat_gain_kw_per_k, cop=cooling_cop)
    total = round(h_kwh + c_kwh, 4)
    return HVACEstimate(
        heating_kwh=h_kwh,
        cooling_kwh=c_kwh,
        total_kwh=total,
        estimated_cost=round(total * energy_rate, 4),
        hdd=hdd,
        cdd=cdd,
    )
