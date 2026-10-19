"""Electricity grid emissions factor lookups and CO2 intensity calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Region(str, Enum):
    """Major grid regions with representative marginal emissions factors."""

    UK = "UK"
    DE = "DE"
    FR = "FR"
    US_WEST = "US_WEST"
    US_EAST = "US_EAST"
    AU_NEM = "AU_NEM"


# kg CO2e per kWh — approximate annual average grid intensities
GRID_INTENSITY_KG_CO2_PER_KWH: dict[Region, float] = {
    Region.UK: 0.233,
    Region.DE: 0.366,
    Region.FR: 0.052,
    Region.US_WEST: 0.290,
    Region.US_EAST: 0.380,
    Region.AU_NEM: 0.650,
}


@dataclass
class EmissionResult:
    """CO2 emissions calculation result."""

    region: str
    energy_kwh: float
    emissions_factor_kg_per_kwh: float
    co2_kg: float
    co2_tonnes: float


def emissions_factor_kg_per_kwh(region: Region) -> float:
    """Return the grid average CO2 intensity for a region.

    Args:
        region: Grid region enum value.

    Returns:
        CO2e intensity in kg/kWh.

    Raises:
        KeyError: If region not found in lookup table.
    """
    return GRID_INTENSITY_KG_CO2_PER_KWH[region]


def calculate_co2(energy_kwh: float, region: Region) -> EmissionResult:
    """Calculate CO2 emissions for a given energy consumption in a region.

    Args:
        energy_kwh: Energy consumed in kWh (must be non-negative).
        region: Grid region for emissions factor lookup.

    Returns:
        :class:`EmissionResult` with CO2 in kg and tonnes.

    Raises:
        ValueError: If energy_kwh is negative.
    """
    if energy_kwh < 0:
        raise ValueError("energy_kwh must be non-negative")
    factor = emissions_factor_kg_per_kwh(region)
    co2_kg = round(energy_kwh * factor, 6)
    return EmissionResult(
        region=region.value,
        energy_kwh=energy_kwh,
        emissions_factor_kg_per_kwh=factor,
        co2_kg=co2_kg,
        co2_tonnes=round(co2_kg / 1000.0, 9),
    )


def compare_regions(energy_kwh: float, regions: list[Region]) -> list[EmissionResult]:
    """Compare CO2 emissions across multiple regions for the same energy use.

    Args:
        energy_kwh: Energy in kWh.
        regions: List of regions to compare.

    Returns:
        List of :class:`EmissionResult` sorted by ascending CO2 intensity.
    """
    results = [calculate_co2(energy_kwh, r) for r in regions]
    return sorted(results, key=lambda r: r.emissions_factor_kg_per_kwh)


def emissions_saved_by_renewables(
    dirty_kwh: float,
    clean_kwh: float,
    region: Region,
) -> float:
    """Estimate CO2 saved by substituting grid power with clean generation.

    Args:
        dirty_kwh: Electricity that would have been drawn from the grid (kWh).
        clean_kwh: Electricity actually supplied by renewables (kWh).
        region: Region for marginal grid emissions factor.

    Returns:
        CO2 saved in kg (rounded to 4 decimal places).  Zero if clean_kwh
        exceeds dirty_kwh.

    Raises:
        ValueError: If either argument is negative.
    """
    if dirty_kwh < 0 or clean_kwh < 0:
        raise ValueError("Energy values must be non-negative")
    displaced_kwh = min(dirty_kwh, clean_kwh)
    factor = emissions_factor_kg_per_kwh(region)
    return round(displaced_kwh * factor, 4)


def annual_budget_check(
    annual_kwh: float,
    region: Region,
    budget_kg_co2: float,
) -> dict[str, float | bool]:
    """Check whether projected annual consumption stays within a CO2 budget.

    Args:
        annual_kwh: Projected annual consumption in kWh.
        region: Grid region.
        budget_kg_co2: Maximum allowed CO2 emissions in kg.

    Returns:
        Dict with keys ``projected_kg``, ``budget_kg``, ``within_budget``,
        ``headroom_kg``.
    """
    result = calculate_co2(annual_kwh, region)
    headroom = budget_kg_co2 - result.co2_kg
    return {
        "projected_kg": result.co2_kg,
        "budget_kg": budget_kg_co2,
        "within_budget": result.co2_kg <= budget_kg_co2,
        "headroom_kg": round(headroom, 4),
    }
