"""Geothermal energy potential and heat extraction modelling."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

EARTH_GEOTHERMAL_GRADIENT_C_PER_KM: float = 25.0
"""Average geothermal gradient in °C/km depth."""

ROCK_THERMAL_CONDUCTIVITY_W_PER_MK: float = 2.5
"""Typical granite thermal conductivity in W/(m·K)."""

WATER_SPECIFIC_HEAT_KJ_PER_KG_K: float = 4.186
"""Specific heat of water in kJ/(kg·K)."""

KJ_PER_KWH: float = 3600.0
"""Kilojoules per kilowatt-hour."""


@dataclass
class GeothermalResource:
    """Characterisation of a geothermal resource."""

    depth_m: float
    temperature_c: float
    surface_temperature_c: float
    thermal_gradient_c_per_m: float
    heat_flow_w_per_m2: float


@dataclass
class GeothermalPlantResult:
    """Operational result for a geothermal power plant."""

    resource_temp_c: float
    rejection_temp_c: float
    carnot_efficiency: float
    thermal_efficiency: float
    thermal_power_kw: float
    electrical_power_kw: float
    annual_mwh: float
    co2_saved_kg: float


def reservoir_temperature(
    depth_m: float,
    surface_temperature_c: float = 15.0,
    gradient_c_per_km: float = EARTH_GEOTHERMAL_GRADIENT_C_PER_KM,
) -> float:
    """Estimate reservoir temperature at a given depth.

    Args:
        depth_m: Depth in metres.
        surface_temperature_c: Mean annual surface temperature in °C.
        gradient_c_per_km: Geothermal gradient in °C/km.

    Returns:
        Estimated reservoir temperature in °C.

    Raises:
        ValueError: If depth is negative.
    """
    if depth_m < 0:
        raise ValueError("depth_m must be non-negative")
    return round(surface_temperature_c + gradient_c_per_km * depth_m / 1000.0, 3)


def heat_flow(
    thermal_conductivity_w_per_mk: float,
    gradient_c_per_m: float,
) -> float:
    """Compute conductive heat flow (Fourier's law).

    Args:
        thermal_conductivity_w_per_mk: Rock thermal conductivity in W/(m·K).
        gradient_c_per_m: Temperature gradient in °C/m.

    Returns:
        Heat flux in W/m².

    Raises:
        ValueError: If conductivity or gradient is non-positive.
    """
    if thermal_conductivity_w_per_mk <= 0:
        raise ValueError("thermal_conductivity must be positive")
    if gradient_c_per_m <= 0:
        raise ValueError("gradient_c_per_m must be positive")
    return round(thermal_conductivity_w_per_mk * gradient_c_per_m, 6)


def carnot_efficiency(
    source_temp_c: float,
    rejection_temp_c: float,
) -> float:
    """Maximum (Carnot) thermal cycle efficiency.

    Args:
        source_temp_c: Heat source temperature in °C.
        rejection_temp_c: Heat rejection (sink) temperature in °C.

    Returns:
        Carnot efficiency as a fraction in [0, 1).

    Raises:
        ValueError: If source temperature is not strictly greater than rejection.
    """
    if source_temp_c <= rejection_temp_c:
        raise ValueError("source_temp_c must be greater than rejection_temp_c")
    t_hot = source_temp_c + 273.15
    t_cold = rejection_temp_c + 273.15
    return round(1.0 - t_cold / t_hot, 6)


def thermal_power_from_flow(
    mass_flow_kg_per_s: float,
    inlet_temp_c: float,
    outlet_temp_c: float,
) -> float:
    """Thermal power extracted from a geothermal fluid flow.

    Args:
        mass_flow_kg_per_s: Brine/fluid mass flow rate in kg/s.
        inlet_temp_c: Fluid temperature entering heat exchanger in °C.
        outlet_temp_c: Fluid temperature leaving heat exchanger in °C.

    Returns:
        Thermal power in kW.

    Raises:
        ValueError: If mass flow is negative or inlet <= outlet temperature.
    """
    if mass_flow_kg_per_s < 0:
        raise ValueError("mass_flow_kg_per_s must be non-negative")
    if inlet_temp_c <= outlet_temp_c:
        raise ValueError("inlet_temp_c must be greater than outlet_temp_c")
    delta_t = inlet_temp_c - outlet_temp_c
    power_kw = mass_flow_kg_per_s * WATER_SPECIFIC_HEAT_KJ_PER_KG_K * delta_t
    return round(power_kw, 4)


def geothermal_plant(
    resource_temp_c: float,
    rejection_temp_c: float = 25.0,
    thermal_power_kw: float = 5000.0,
    plant_efficiency_fraction: float = 0.10,
    capacity_factor: float = 0.90,
) -> GeothermalPlantResult:
    """Model a geothermal binary-cycle or flash power plant.

    Args:
        resource_temp_c: Geothermal resource temperature in °C.
        rejection_temp_c: Condenser rejection temperature in °C.
        thermal_power_kw: Thermal input power in kW.
        plant_efficiency_fraction: Fraction of Carnot achieved by the real plant.
        capacity_factor: Plant capacity factor (0-1).

    Returns:
        Populated :class:`GeothermalPlantResult`.

    Raises:
        ValueError: If resource temperature <= rejection, thermal power <= 0.
    """
    if resource_temp_c <= rejection_temp_c:
        raise ValueError("resource_temp_c must be greater than rejection_temp_c")
    if thermal_power_kw <= 0:
        raise ValueError("thermal_power_kw must be positive")
    if not (0 < plant_efficiency_fraction <= 1):
        raise ValueError("plant_efficiency_fraction must be in (0, 1]")

    eta_carnot = carnot_efficiency(resource_temp_c, rejection_temp_c)
    eta_thermal = eta_carnot * plant_efficiency_fraction
    electrical_kw = thermal_power_kw * eta_thermal
    annual_mwh = electrical_kw * 8760 * capacity_factor / 1000.0
    co2_saved_kg = annual_mwh * 1000 * 0.233

    return GeothermalPlantResult(
        resource_temp_c=resource_temp_c,
        rejection_temp_c=rejection_temp_c,
        carnot_efficiency=round(eta_carnot, 6),
        thermal_efficiency=round(eta_thermal, 6),
        thermal_power_kw=round(thermal_power_kw, 4),
        electrical_power_kw=round(electrical_kw, 4),
        annual_mwh=round(annual_mwh, 4),
        co2_saved_kg=round(co2_saved_kg, 2),
    )


def characterise_resource(
    depth_m: float,
    surface_temperature_c: float = 15.0,
    gradient_c_per_km: float = EARTH_GEOTHERMAL_GRADIENT_C_PER_KM,
    conductivity: float = ROCK_THERMAL_CONDUCTIVITY_W_PER_MK,
) -> GeothermalResource:
    """Build a :class:`GeothermalResource` descriptor for a site.

    Args:
        depth_m: Target depth in metres.
        surface_temperature_c: Mean surface temperature in °C.
        gradient_c_per_km: Geothermal gradient in °C/km.
        conductivity: Rock thermal conductivity in W/(m·K).

    Returns:
        Populated :class:`GeothermalResource`.
    """
    temp = reservoir_temperature(depth_m, surface_temperature_c, gradient_c_per_km)
    gradient_per_m = gradient_c_per_km / 1000.0
    hf = heat_flow(conductivity, gradient_per_m)
    return GeothermalResource(
        depth_m=depth_m,
        temperature_c=temp,
        surface_temperature_c=surface_temperature_c,
        thermal_gradient_c_per_m=round(gradient_per_m, 6),
        heat_flow_w_per_m2=hf,
    )
