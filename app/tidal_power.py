"""Tidal stream and tidal barrage energy estimation utilities."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SEA_WATER_DENSITY_KG_PER_M3: float = 1025.0
"""Typical density of sea water in kg/m³."""

GRAVITATIONAL_ACC: float = 9.81
"""Standard gravitational acceleration in m/s²."""

BETZ_LIMIT_TIDAL: float = 16.0 / 27.0
"""Betz limit — theoretical maximum power extraction coefficient."""

KWH_PER_MWH: float = 1000.0
"""kWh per MWh."""


@dataclass
class TidalStreamResult:
    """Output of a tidal stream turbine calculation."""

    rotor_area_m2: float
    theoretical_power_kw: float
    actual_power_kw: float
    capacity_factor: float
    annual_mwh: float
    co2_saved_kg: float


@dataclass
class BarrageResult:
    """Output of a tidal barrage energy calculation."""

    basin_area_m2: float
    tidal_range_m: float
    potential_energy_kwh: float
    electrical_energy_kwh: float
    cycles_per_day: float
    annual_mwh: float


def tidal_stream_power(
    rotor_diameter_m: float,
    current_speed_m_per_s: float,
    power_coefficient: float = 0.35,
    sea_water_density: float = SEA_WATER_DENSITY_KG_PER_M3,
) -> TidalStreamResult:
    """Estimate tidal stream turbine power output.

    Uses the standard kinetic energy formula: P = 0.5 * rho * Cp * A * v³.

    Args:
        rotor_diameter_m: Turbine rotor diameter in metres.
        current_speed_m_per_s: Free-stream tidal current speed in m/s.
        power_coefficient: Cp (default 0.35, below Betz limit).
        sea_water_density: Density of sea water in kg/m³.

    Returns:
        Populated :class:`TidalStreamResult`.

    Raises:
        ValueError: If rotor diameter or current speed is non-positive, or Cp > Betz limit.
    """
    if rotor_diameter_m <= 0:
        raise ValueError("rotor_diameter_m must be positive")
    if current_speed_m_per_s < 0:
        raise ValueError("current_speed_m_per_s must be non-negative")
    if power_coefficient > BETZ_LIMIT_TIDAL:
        raise ValueError(f"power_coefficient cannot exceed Betz limit ({BETZ_LIMIT_TIDAL:.4f})")

    area = math.pi * (rotor_diameter_m / 2.0) ** 2
    theoretical_kw = 0.5 * sea_water_density * area * current_speed_m_per_s**3 / 1000.0
    actual_kw = theoretical_kw * power_coefficient / BETZ_LIMIT_TIDAL * BETZ_LIMIT_TIDAL
    actual_kw = (
        0.5 * sea_water_density * area * power_coefficient * current_speed_m_per_s**3 / 1000.0
    )
    capacity_factor = actual_kw / theoretical_kw if theoretical_kw > 0 else 0.0
    annual_mwh = actual_kw * 8760 / KWH_PER_MWH
    co2_saved_kg = annual_mwh * 1000 * 0.233

    return TidalStreamResult(
        rotor_area_m2=round(area, 4),
        theoretical_power_kw=round(theoretical_kw, 4),
        actual_power_kw=round(actual_kw, 4),
        capacity_factor=round(capacity_factor, 4),
        annual_mwh=round(annual_mwh, 4),
        co2_saved_kg=round(co2_saved_kg, 2),
    )


def tidal_barrage_energy(
    basin_area_m2: float,
    tidal_range_m: float,
    turbine_efficiency: float = 0.85,
    generator_efficiency: float = 0.95,
    tidal_cycles_per_day: float = 2.0,
) -> BarrageResult:
    """Estimate energy from a tidal barrage over one day.

    Args:
        basin_area_m2: Surface area of the tidal basin in m².
        tidal_range_m: Tidal range (peak-to-trough height) in metres.
        turbine_efficiency: Hydraulic turbine efficiency in (0, 1].
        generator_efficiency: Generator efficiency in (0, 1].
        tidal_cycles_per_day: Number of tidal cycles per day (default 2 for semi-diurnal).

    Returns:
        Populated :class:`BarrageResult`.

    Raises:
        ValueError: If basin area or tidal range is non-positive or efficiencies out of range.
    """
    if basin_area_m2 <= 0:
        raise ValueError("basin_area_m2 must be positive")
    if tidal_range_m <= 0:
        raise ValueError("tidal_range_m must be positive")
    if not (0 < turbine_efficiency <= 1):
        raise ValueError("turbine_efficiency must be in (0, 1]")
    if not (0 < generator_efficiency <= 1):
        raise ValueError("generator_efficiency must be in (0, 1]")

    head = tidal_range_m / 2.0
    volume_m3 = basin_area_m2 * head
    potential_j_per_cycle = SEA_WATER_DENSITY_KG_PER_M3 * GRAVITATIONAL_ACC * volume_m3 * head / 2.0
    potential_kwh_per_cycle = potential_j_per_cycle / 3.6e6
    electrical_kwh_per_cycle = potential_kwh_per_cycle * turbine_efficiency * generator_efficiency
    total_potential_kwh = potential_kwh_per_cycle * tidal_cycles_per_day
    total_electrical_kwh = electrical_kwh_per_cycle * tidal_cycles_per_day
    annual_mwh = total_electrical_kwh * 365 / KWH_PER_MWH

    return BarrageResult(
        basin_area_m2=basin_area_m2,
        tidal_range_m=tidal_range_m,
        potential_energy_kwh=round(total_potential_kwh, 4),
        electrical_energy_kwh=round(total_electrical_kwh, 4),
        cycles_per_day=tidal_cycles_per_day,
        annual_mwh=round(annual_mwh, 4),
    )


def current_power_density(
    current_speed_m_per_s: float,
    sea_water_density: float = SEA_WATER_DENSITY_KG_PER_M3,
) -> float:
    """Power density of a tidal current in W/m².

    Args:
        current_speed_m_per_s: Tidal current speed in m/s.
        sea_water_density: Density of sea water in kg/m³.

    Returns:
        Kinetic power density in W/m².

    Raises:
        ValueError: If current speed is negative.
    """
    if current_speed_m_per_s < 0:
        raise ValueError("current_speed_m_per_s must be non-negative")
    return round(0.5 * sea_water_density * current_speed_m_per_s**3, 4)


def minimum_cut_in_speed(
    rated_power_kw: float,
    rotor_diameter_m: float,
    power_coefficient: float = 0.35,
    sea_water_density: float = SEA_WATER_DENSITY_KG_PER_M3,
) -> float:
    """Minimum current speed to achieve rated power.

    Args:
        rated_power_kw: Turbine rated power in kW.
        rotor_diameter_m: Rotor diameter in metres.
        power_coefficient: Power coefficient Cp.
        sea_water_density: Sea water density in kg/m³.

    Returns:
        Cut-in current speed in m/s.

    Raises:
        ValueError: If rated_power_kw or rotor_diameter_m are non-positive.
    """
    if rated_power_kw <= 0:
        raise ValueError("rated_power_kw must be positive")
    if rotor_diameter_m <= 0:
        raise ValueError("rotor_diameter_m must be positive")
    area = math.pi * (rotor_diameter_m / 2.0) ** 2
    v_cubed = (rated_power_kw * 1000) / (0.5 * sea_water_density * power_coefficient * area)
    return round(v_cubed ** (1.0 / 3.0), 4)


def tidal_energy_density(current_speed_m_s: float, seawater_density: float = 1025.0) -> float:
    """Return the kinetic energy density of tidal flow in J/m³.

    Args:
        current_speed_m_s: Tidal current speed in m/s (non-negative).
        seawater_density: Seawater density in kg/m³ (default 1025).

    Returns:
        Kinetic energy density in J/m³.

    Raises:
        ValueError: If current_speed_m_s is negative.
    """
    if current_speed_m_s < 0:
        raise ValueError("current_speed_m_s must be non-negative")
    return 0.5 * seawater_density * current_speed_m_s**2


def annual_tidal_energy_kwh(
    rated_power_kw: float,
    capacity_factor: float = 0.35,
) -> float:
    """Estimate annual energy output from a tidal device.

    Args:
        rated_power_kw: Rated power capacity in kW.
        capacity_factor: Expected capacity factor (0 < cf ≤ 1, default 0.35).

    Returns:
        Annual energy in kWh.

    Raises:
        ValueError: If rated_power_kw ≤ 0 or capacity_factor not in (0, 1].
    """
    if rated_power_kw <= 0:
        raise ValueError("rated_power_kw must be positive")
    if not 0 < capacity_factor <= 1.0:
        raise ValueError("capacity_factor must be in (0, 1]")
    return rated_power_kw * 8760.0 * capacity_factor


def tidal_resource_score(
    mean_speed_m_s: float,
    cut_in_speed_m_s: float = 0.5,
    rated_speed_m_s: float = 2.5,
) -> float:
    """Return a 0–100 score for a tidal resource suitability.

    Score is proportional to how far the mean speed exceeds the cut-in speed
    relative to the rated speed range.

    Args:
        mean_speed_m_s: Mean tidal current speed in m/s.
        cut_in_speed_m_s: Minimum speed for power generation in m/s.
        rated_speed_m_s: Speed at which rated power is reached in m/s.

    Returns:
        Resource score in [0, 100].

    Raises:
        ValueError: If rated_speed_m_s ≤ cut_in_speed_m_s or mean_speed_m_s < 0.
    """
    if mean_speed_m_s < 0:
        raise ValueError("mean_speed_m_s must be non-negative")
    if rated_speed_m_s <= cut_in_speed_m_s:
        raise ValueError("rated_speed_m_s must exceed cut_in_speed_m_s")
    if mean_speed_m_s < cut_in_speed_m_s:
        return 0.0
    score = min((mean_speed_m_s - cut_in_speed_m_s) / (rated_speed_m_s - cut_in_speed_m_s), 1.0)
    return round(score * 100.0, 2)
