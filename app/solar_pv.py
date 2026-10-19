"""Solar PV generation estimation and performance analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

STANDARD_TEST_CONDITION_IRRADIANCE: float = 1000.0
"""W/m² irradiance at Standard Test Conditions (STC)."""

STANDARD_TEST_CONDITION_TEMP: float = 25.0
"""°C cell temperature at STC."""


@dataclass
class PVSystemResult:
    """Result of a PV system performance calculation."""

    peak_power_kw: float
    annual_yield_kwh: float
    performance_ratio: float
    specific_yield_kwh_per_kwp: float
    capacity_factor: float
    co2_saved_kg: float


def array_power_kw(
    panel_count: int,
    panel_watt_peak: float,
    temperature_c: float,
    irradiance_w_per_m2: float,
    temp_coefficient_pct_per_c: float = -0.40,
) -> float:
    """Estimate instantaneous DC output of a PV array.

    Args:
        panel_count: Number of PV panels.
        panel_watt_peak: Nameplate peak power per panel in Wp.
        temperature_c: Ambient cell temperature in °C.
        irradiance_w_per_m2: Plane-of-array irradiance in W/m².
        temp_coefficient_pct_per_c: Power temperature coefficient (%/°C).

    Returns:
        Array DC power in kW.

    Raises:
        ValueError: If any argument is non-positive or irradiance is negative.
    """
    if panel_count <= 0:
        raise ValueError("panel_count must be positive")
    if panel_watt_peak <= 0:
        raise ValueError("panel_watt_peak must be positive")
    if irradiance_w_per_m2 < 0:
        raise ValueError("irradiance_w_per_m2 must be non-negative")
    stc_power_w = panel_count * panel_watt_peak
    temp_derating = 1.0 + (temp_coefficient_pct_per_c / 100.0) * (
        temperature_c - STANDARD_TEST_CONDITION_TEMP
    )
    irradiance_factor = irradiance_w_per_m2 / STANDARD_TEST_CONDITION_IRRADIANCE
    return round(stc_power_w * temp_derating * irradiance_factor / 1000.0, 4)


def annual_yield_kwh(
    peak_kw: float,
    peak_sun_hours: float,
    performance_ratio: float = 0.75,
) -> float:
    """Estimate annual AC energy yield from a PV system.

    Args:
        peak_kw: Installed DC capacity in kWp.
        peak_sun_hours: Average annual peak sun hours per day.
        performance_ratio: System efficiency factor (0–1).

    Returns:
        Estimated annual AC yield in kWh.

    Raises:
        ValueError: If arguments are out of valid range.
    """
    if peak_kw <= 0:
        raise ValueError("peak_kw must be positive")
    if peak_sun_hours < 0:
        raise ValueError("peak_sun_hours must be non-negative")
    if not (0 < performance_ratio <= 1):
        raise ValueError("performance_ratio must be in (0, 1]")
    return round(peak_kw * peak_sun_hours * 365 * performance_ratio, 2)


def capacity_factor(
    annual_kwh_produced: float,
    installed_kw: float,
) -> float:
    """Compute annual capacity factor for a PV system.

    Args:
        annual_kwh_produced: Measured annual generation in kWh.
        installed_kw: Installed DC capacity in kWp.

    Returns:
        Capacity factor as a fraction (0–1).

    Raises:
        ValueError: If arguments are non-positive.
    """
    if annual_kwh_produced < 0:
        raise ValueError("annual_kwh_produced must be non-negative")
    if installed_kw <= 0:
        raise ValueError("installed_kw must be positive")
    return round(annual_kwh_produced / (installed_kw * 8760), 4)


def specific_yield(annual_kwh_produced: float, installed_kwp: float) -> float:
    """Compute specific yield (kWh per kWp installed).

    Args:
        annual_kwh_produced: Annual AC generation in kWh.
        installed_kwp: Installed DC capacity in kWp.

    Returns:
        Specific yield in kWh/kWp.

    Raises:
        ValueError: If arguments are non-positive.
    """
    if annual_kwh_produced < 0:
        raise ValueError("annual_kwh_produced must be non-negative")
    if installed_kwp <= 0:
        raise ValueError("installed_kwp must be positive")
    return round(annual_kwh_produced / installed_kwp, 2)


def analyse_pv_system(
    panel_count: int,
    panel_watt_peak: float,
    peak_sun_hours: float,
    performance_ratio: float = 0.75,
    grid_emission_factor_kg_per_kwh: float = 0.233,
) -> PVSystemResult:
    """Run a complete annual performance analysis for a PV system.

    Args:
        panel_count: Number of PV panels.
        panel_watt_peak: Nameplate peak power per panel in Wp.
        peak_sun_hours: Average daily peak sun hours.
        performance_ratio: System performance ratio.
        grid_emission_factor_kg_per_kwh: Grid emission factor for CO₂ savings.

    Returns:
        Populated :class:`PVSystemResult`.
    """
    peak_kw = panel_count * panel_watt_peak / 1000.0
    yield_kwh = annual_yield_kwh(peak_kw, peak_sun_hours, performance_ratio)
    cf = capacity_factor(yield_kwh, peak_kw)
    sy = specific_yield(yield_kwh, peak_kw)
    co2 = round(yield_kwh * grid_emission_factor_kg_per_kwh, 2)
    return PVSystemResult(
        peak_power_kw=round(peak_kw, 3),
        annual_yield_kwh=yield_kwh,
        performance_ratio=performance_ratio,
        specific_yield_kwh_per_kwp=sy,
        capacity_factor=cf,
        co2_saved_kg=co2,
    )
