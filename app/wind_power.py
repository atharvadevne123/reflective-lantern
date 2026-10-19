"""Wind power estimation and turbine performance modelling."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

AIR_DENSITY_SEA_LEVEL: float = 1.225
"""Standard air density at sea level in kg/m³."""

BETZ_LIMIT: float = 16.0 / 27.0
"""Theoretical maximum power coefficient (Betz limit ≈ 0.593)."""


@dataclass
class TurbineResult:
    """Performance result for a wind turbine."""

    rotor_area_m2: float
    theoretical_power_kw: float
    actual_power_kw: float
    capacity_factor: float
    annual_kwh: float
    co2_saved_kg: float


def rotor_swept_area(rotor_diameter_m: float) -> float:
    """Compute the swept area of a horizontal-axis wind turbine rotor.

    Args:
        rotor_diameter_m: Rotor diameter in metres.

    Returns:
        Swept area in m².

    Raises:
        ValueError: If diameter is non-positive.
    """
    if rotor_diameter_m <= 0:
        raise ValueError("rotor_diameter_m must be positive")
    return round(math.pi * (rotor_diameter_m / 2.0) ** 2, 4)


def wind_power_density(
    wind_speed_m_per_s: float,
    air_density_kg_per_m3: float = AIR_DENSITY_SEA_LEVEL,
) -> float:
    """Compute wind power density in W/m².

    Args:
        wind_speed_m_per_s: Wind speed in m/s.
        air_density_kg_per_m3: Air density in kg/m³ (default sea-level).

    Returns:
        Wind power density in W/m².

    Raises:
        ValueError: If wind speed is negative or air density is non-positive.
    """
    if wind_speed_m_per_s < 0:
        raise ValueError("wind_speed_m_per_s must be non-negative")
    if air_density_kg_per_m3 <= 0:
        raise ValueError("air_density_kg_per_m3 must be positive")
    return round(0.5 * air_density_kg_per_m3 * wind_speed_m_per_s**3, 4)


def turbine_power_kw(
    wind_speed_m_per_s: float,
    rotor_diameter_m: float,
    power_coefficient: float = 0.40,
    air_density_kg_per_m3: float = AIR_DENSITY_SEA_LEVEL,
) -> float:
    """Estimate instantaneous turbine output power in kW.

    Args:
        wind_speed_m_per_s: Hub-height wind speed in m/s.
        rotor_diameter_m: Rotor diameter in metres.
        power_coefficient: Turbine power coefficient Cp (default 0.40).
        air_density_kg_per_m3: Air density in kg/m³.

    Returns:
        Turbine output power in kW.

    Raises:
        ValueError: If any argument is out of valid range.
    """
    if wind_speed_m_per_s < 0:
        raise ValueError("wind_speed_m_per_s must be non-negative")
    if rotor_diameter_m <= 0:
        raise ValueError("rotor_diameter_m must be positive")
    if not (0 < power_coefficient <= BETZ_LIMIT + 0.01):
        raise ValueError(f"power_coefficient must be in (0, {BETZ_LIMIT + 0.01:.3f}]")
    area = rotor_swept_area(rotor_diameter_m)
    density = wind_power_density(wind_speed_m_per_s, air_density_kg_per_m3)
    return round(density * area * power_coefficient / 1000.0, 4)


def annual_energy_output_kwh(
    rated_kw: float,
    capacity_factor: float,
) -> float:
    """Estimate annual energy output from rated power and capacity factor.

    Args:
        rated_kw: Turbine rated (nameplate) power in kW.
        capacity_factor: Annual capacity factor as a fraction (0–1).

    Returns:
        Annual energy output in kWh.

    Raises:
        ValueError: If arguments are out of valid range.
    """
    if rated_kw <= 0:
        raise ValueError("rated_kw must be positive")
    if not (0 < capacity_factor <= 1):
        raise ValueError("capacity_factor must be in (0, 1]")
    return round(rated_kw * capacity_factor * 8760.0, 2)


def wind_speed_at_height(
    reference_speed: float,
    reference_height: float,
    target_height: float,
    hellmann_exponent: float = 1.0 / 7.0,
) -> float:
    """Estimate wind speed at a target height using the power law profile.

    Args:
        reference_speed: Wind speed at the reference height in m/s.
        reference_height: Reference height in metres.
        target_height: Target hub height in metres.
        hellmann_exponent: Hellmann (shear) exponent (default 1/7 for open terrain).

    Returns:
        Estimated wind speed at target height in m/s.

    Raises:
        ValueError: If heights or speed are non-positive.
    """
    if reference_speed < 0:
        raise ValueError("reference_speed must be non-negative")
    if reference_height <= 0 or target_height <= 0:
        raise ValueError("Heights must be positive")
    return round(reference_speed * (target_height / reference_height) ** hellmann_exponent, 4)


def analyse_turbine(
    rotor_diameter_m: float,
    rated_kw: float,
    mean_wind_speed_m_per_s: float,
    power_coefficient: float = 0.40,
    capacity_factor: float = 0.30,
    grid_emission_factor_kg_per_kwh: float = 0.233,
) -> TurbineResult:
    """Run a complete annual performance analysis for a wind turbine.

    Args:
        rotor_diameter_m: Rotor diameter in metres.
        rated_kw: Turbine rated capacity in kW.
        mean_wind_speed_m_per_s: Mean annual hub-height wind speed in m/s.
        power_coefficient: Power coefficient Cp.
        capacity_factor: Expected annual capacity factor.
        grid_emission_factor_kg_per_kwh: Grid intensity for CO₂ savings.

    Returns:
        Populated :class:`TurbineResult`.
    """
    area = rotor_swept_area(rotor_diameter_m)
    theoretical_kw = turbine_power_kw(mean_wind_speed_m_per_s, rotor_diameter_m, BETZ_LIMIT)
    actual_kw = turbine_power_kw(mean_wind_speed_m_per_s, rotor_diameter_m, power_coefficient)
    kwh = annual_energy_output_kwh(rated_kw, capacity_factor)
    co2 = round(kwh * grid_emission_factor_kg_per_kwh, 2)
    return TurbineResult(
        rotor_area_m2=area,
        theoretical_power_kw=theoretical_kw,
        actual_power_kw=actual_kw,
        capacity_factor=capacity_factor,
        annual_kwh=kwh,
        co2_saved_kg=co2,
    )
