"""Carbon footprint estimation from energy consumption data."""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)

# Grid carbon intensity factors (kg CO2e per kWh) by region
GRID_CARBON_INTENSITY: dict[str, float] = {
    "northeast": 0.25,
    "midwest": 0.45,
    "south": 0.38,
    "west": 0.22,
    "texas": 0.40,
    "pacific_nw": 0.10,
    "new_england": 0.24,
    "mountain": 0.35,
    "southeast": 0.36,
    "florida": 0.37,
    "default": 0.35,
}

KG_TO_TONNES = 0.001
TREES_PER_TONNE_CO2_PER_YEAR = 50.0


@functools.lru_cache(maxsize=32)
def _grid_intensity(region: str) -> float:
    """Return cached grid carbon intensity for *region* (kg CO2e/kWh)."""
    return GRID_CARBON_INTENSITY.get(region, GRID_CARBON_INTENSITY["default"])


def kwh_to_co2_kg(kwh: float, region: str = "default") -> float:
    """Convert energy consumption (kWh) to CO2 equivalent in kilograms.

    Args:
        kwh: Energy consumption in kilowatt-hours.
        region: Grid region identifier (case-insensitive); falls back to 'default'.

    Returns:
        Estimated CO2 equivalent in kilograms, rounded to 4 decimal places.

    Raises:
        ValueError: If *kwh* is negative.
    """
    if kwh < 0:
        raise ValueError(f"kwh must be non-negative, got {kwh}")
    intensity = _grid_intensity(region.lower())
    result = kwh * intensity
    logger.debug("CO2 estimate: %.3f kWh x %.4f kg/kWh = %.4f kg CO2e", kwh, intensity, result)
    return round(result, 4)


def co2_kg_to_tonnes(kg: float) -> float:
    """Convert kilograms of CO2 to metric tonnes.

    Args:
        kg: CO2 equivalent in kilograms.

    Returns:
        CO2 equivalent in metric tonnes, rounded to 6 decimal places.
    """
    return round(kg * KG_TO_TONNES, 6)


def trees_equivalent(co2_kg: float) -> float:
    """Estimate equivalent number of trees needed to absorb given CO2 in one year.

    Uses the approximation that one mature tree absorbs ~20 kg CO2/year,
    which equals 50 trees per metric tonne.

    Args:
        co2_kg: CO2 equivalent in kilograms.

    Returns:
        Estimated number of trees, rounded to 1 decimal place.
    """
    tonnes = co2_kg_to_tonnes(co2_kg)
    return round(tonnes * TREES_PER_TONNE_CO2_PER_YEAR, 1)


def annual_carbon_report(
    monthly_kwh: list[float],
    region: str = "default",
) -> dict[str, object]:
    """Generate an annual carbon footprint report from monthly consumption.

    Args:
        monthly_kwh: List of 12 monthly energy consumption values in kWh.
        region: Grid region identifier.

    Returns:
        Dict with total_kwh, total_co2_kg, total_co2_tonnes, trees_equivalent,
        and monthly_co2_kg list.

    Raises:
        ValueError: If *monthly_kwh* does not have exactly 12 elements.
    """
    if len(monthly_kwh) != 12:
        raise ValueError(f"monthly_kwh must have exactly 12 elements, got {len(monthly_kwh)}")
    monthly_co2 = [kwh_to_co2_kg(kwh, region) for kwh in monthly_kwh]
    total_kwh = sum(monthly_kwh)
    total_co2_kg = sum(monthly_co2)
    total_co2_tonnes = co2_kg_to_tonnes(total_co2_kg)
    return {
        "region": region,
        "total_kwh": round(total_kwh, 2),
        "total_co2_kg": round(total_co2_kg, 2),
        "total_co2_tonnes": total_co2_tonnes,
        "trees_equivalent": trees_equivalent(total_co2_kg),
        "monthly_co2_kg": [round(v, 2) for v in monthly_co2],
    }


def carbon_savings(
    actual_kwh: float,
    baseline_kwh: float,
    region: str = "default",
) -> dict[str, float]:
    """Estimate CO2 savings from reduced energy consumption.

    Args:
        actual_kwh: Measured energy consumption in kWh.
        baseline_kwh: Reference baseline consumption in kWh.
        region: Grid region identifier.

    Returns:
        Dict with saved_kwh, saved_co2_kg, saved_co2_tonnes, and trees_saved.
    """
    saved_kwh = max(0.0, baseline_kwh - actual_kwh)
    saved_co2_kg = kwh_to_co2_kg(saved_kwh, region)
    return {
        "saved_kwh": round(saved_kwh, 3),
        "saved_co2_kg": round(saved_co2_kg, 4),
        "saved_co2_tonnes": co2_kg_to_tonnes(saved_co2_kg),
        "trees_saved": trees_equivalent(saved_co2_kg),
    }

__all__ = [
    "GRID_CARBON_INTENSITY",
    "KG_TO_TONNES",
    "TREES_PER_TONNE_CO2_PER_YEAR",
    "annual_carbon_report",
    "carbon_savings",
    "co2_kg_to_tonnes",
    "kwh_to_co2_kg",
    "trees_equivalent",
]
