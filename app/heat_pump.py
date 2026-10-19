"""Heat pump performance modelling and coefficient of performance calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class HeatPumpType(str, Enum):
    """Classification of heat pump system types."""

    AIR_SOURCE = "air_source"
    GROUND_SOURCE = "ground_source"
    WATER_SOURCE = "water_source"


@dataclass
class HeatPumpResult:
    """Performance result for a heat pump system."""

    cop: float
    heat_delivered_kwh: float
    electricity_consumed_kwh: float
    running_cost: float
    co2_emissions_kg: float
    scop: float


def carnot_cop(
    source_temp_c: float,
    sink_temp_c: float,
) -> float:
    """Compute the theoretical Carnot COP for a heat pump.

    Args:
        source_temp_c: Heat source temperature in °C.
        sink_temp_c: Heat sink (delivery) temperature in °C.

    Returns:
        Carnot COP (dimensionless).

    Raises:
        ValueError: If sink temperature is not higher than source temperature.
    """
    if sink_temp_c <= source_temp_c:
        raise ValueError("sink_temp_c must exceed source_temp_c")
    t_source_k = source_temp_c + 273.15
    t_sink_k = sink_temp_c + 273.15
    return round(t_sink_k / (t_sink_k - t_source_k), 4)


def realistic_cop(
    source_temp_c: float,
    sink_temp_c: float,
    efficiency_factor: float = 0.50,
) -> float:
    """Estimate realistic COP as a fraction of Carnot COP.

    Args:
        source_temp_c: Source temperature in °C.
        sink_temp_c: Delivery temperature in °C.
        efficiency_factor: Fraction of Carnot COP achieved (default 50%).

    Returns:
        Estimated real-world COP.

    Raises:
        ValueError: If efficiency_factor is outside (0, 1] or temperatures invalid.
    """
    if not (0 < efficiency_factor <= 1):
        raise ValueError("efficiency_factor must be in (0, 1]")
    return round(carnot_cop(source_temp_c, sink_temp_c) * efficiency_factor, 4)


def heat_delivered_kwh(
    electricity_kwh: float,
    cop: float,
) -> float:
    """Compute thermal energy delivered for given electricity input and COP.

    Args:
        electricity_kwh: Electricity consumed in kWh.
        cop: Coefficient of performance.

    Returns:
        Heat delivered in kWh.

    Raises:
        ValueError: If arguments are non-positive.
    """
    if electricity_kwh < 0:
        raise ValueError("electricity_kwh must be non-negative")
    if cop <= 0:
        raise ValueError("cop must be positive")
    return round(electricity_kwh * cop, 4)


def electricity_for_heat(
    heat_kwh: float,
    cop: float,
) -> float:
    """Compute electricity required to deliver a target heat output.

    Args:
        heat_kwh: Required heat output in kWh.
        cop: Coefficient of performance.

    Returns:
        Electricity consumed in kWh.

    Raises:
        ValueError: If arguments are non-positive.
    """
    if heat_kwh < 0:
        raise ValueError("heat_kwh must be non-negative")
    if cop <= 0:
        raise ValueError("cop must be positive")
    return round(heat_kwh / cop, 4)


def seasonal_cop(
    monthly_cops: list[float],
    monthly_heat_kwh: list[float],
) -> float:
    """Compute the Seasonal Coefficient of Performance (SCOP).

    Args:
        monthly_cops: COP value for each month (12 values).
        monthly_heat_kwh: Heat delivered each month in kWh (12 values).

    Returns:
        Annual SCOP as total heat output / total electricity consumed.

    Raises:
        ValueError: If lists are not the same length, or contain invalid values.
    """
    if len(monthly_cops) != len(monthly_heat_kwh):
        raise ValueError("monthly_cops and monthly_heat_kwh must have the same length")
    if any(c <= 0 for c in monthly_cops):
        raise ValueError("All COP values must be positive")
    if any(h < 0 for h in monthly_heat_kwh):
        raise ValueError("All heat values must be non-negative")
    total_heat = sum(monthly_heat_kwh)
    total_elec = sum(h / c for h, c in zip(monthly_heat_kwh, monthly_cops))
    if total_elec == 0:
        return 0.0
    return round(total_heat / total_elec, 4)


def analyse_heat_pump(
    source_temp_c: float,
    flow_temp_c: float,
    annual_heat_kwh: float,
    efficiency_factor: float = 0.50,
    electricity_rate: float = 0.28,
    grid_emission_factor: float = 0.233,
) -> HeatPumpResult:
    """Run a complete heat pump performance analysis.

    Args:
        source_temp_c: Heat source temperature in °C.
        flow_temp_c: Heat delivery (flow) temperature in °C.
        annual_heat_kwh: Annual heat demand in kWh.
        efficiency_factor: Fraction of Carnot COP achieved.
        electricity_rate: Electricity unit cost in $/kWh.
        grid_emission_factor: Grid emission intensity in kg CO₂/kWh.

    Returns:
        Populated :class:`HeatPumpResult`.
    """
    cop = realistic_cop(source_temp_c, flow_temp_c, efficiency_factor)
    elec_kwh = electricity_for_heat(annual_heat_kwh, cop)
    cost = round(elec_kwh * electricity_rate, 2)
    co2 = round(elec_kwh * grid_emission_factor, 2)
    monthly_cops = [cop] * 12
    monthly_heat = [annual_heat_kwh / 12.0] * 12
    scop = seasonal_cop(monthly_cops, monthly_heat)
    return HeatPumpResult(
        cop=cop,
        heat_delivered_kwh=annual_heat_kwh,
        electricity_consumed_kwh=elec_kwh,
        running_cost=cost,
        co2_emissions_kg=co2,
        scop=scop,
    )
