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


def daily_carbon_estimate(
    hourly_kwh: list[float],
    region: str = "default",
) -> dict[str, float]:
    """Estimate carbon footprint for a single day from hourly consumption.

    Args:
        hourly_kwh: List of up to 24 hourly energy readings in kWh.
        region: Grid region identifier.

    Returns:
        Dict with total_kwh, total_co2_kg, and trees_equivalent for the day.

    Raises:
        ValueError: If *hourly_kwh* contains more than 24 elements.
    """
    if len(hourly_kwh) > 24:
        raise ValueError(f"hourly_kwh must have at most 24 elements, got {len(hourly_kwh)}")
    total_kwh = sum(hourly_kwh)
    total_co2_kg = kwh_to_co2_kg(total_kwh, region)
    return {
        "total_kwh": round(total_kwh, 4),
        "total_co2_kg": total_co2_kg,
        "trees_equivalent": trees_equivalent(total_co2_kg),
    }


__all__ = [
    "GRID_CARBON_INTENSITY",
    "KG_TO_TONNES",
    "TREES_PER_TONNE_CO2_PER_YEAR",
    "annual_carbon_report",
    "carbon_budget_remaining",
    "carbon_intensity_by_hour",
    "carbon_saved_kwh",
    "carbon_savings",
    "co2_kg_to_tonnes",
    "compare_regions",
    "daily_carbon_estimate",
    "kwh_to_co2_kg",
    "monthly_co2_breakdown",
    "tree_offset_days",
    "trees_equivalent",
]


def carbon_intensity_by_hour(
    hourly_kwh: list[float],
    region: str = "default",
) -> list[float]:
    """Return per-hour CO2 emissions (kg) from a list of hourly energy readings.

    Args:
        hourly_kwh: List of hourly energy consumption values in kWh.
        region: Grid region identifier.

    Returns:
        List of CO2 emissions in kg for each hour, same length as *hourly_kwh*.
    """
    return [kwh_to_co2_kg(kwh, region) for kwh in hourly_kwh]


def tree_offset_days(co2_kg: float, num_trees: int = 1) -> float:
    """Estimate how many days *num_trees* need to offset *co2_kg* of emissions.

    Uses the approximation that a mature tree sequesters ~20 kg CO2/year.

    Args:
        co2_kg: CO2 equivalent in kilograms to offset.
        num_trees: Number of trees contributing to the offset.

    Returns:
        Number of days to offset the emissions, rounded to 2 decimal places.
        Returns 0.0 for zero or negative CO2.

    Raises:
        ValueError: If *num_trees* is less than 1.
    """
    if num_trees < 1:
        raise ValueError(f"num_trees must be at least 1, got {num_trees}")
    if co2_kg <= 0:
        return 0.0
    kg_per_tree_per_day = 20.0 / 365.0
    return round(co2_kg / (num_trees * kg_per_tree_per_day), 2)


def monthly_co2_breakdown(
    monthly_kwh: list[float],
    region: str = "default",
) -> list[dict[str, float]]:
    """Return a month-by-month CO2 breakdown from a list of monthly kWh values.

    Args:
        monthly_kwh: List of monthly energy consumption values in kWh (any length).
        region: Grid region identifier.

    Returns:
        List of dicts with 'month' (1-indexed), 'kwh', and 'co2_kg' for each entry.
    """
    return [
        {
            "month": i + 1,
            "kwh": round(kwh, 4),
            "co2_kg": kwh_to_co2_kg(kwh, region),
        }
        for i, kwh in enumerate(monthly_kwh)
    ]


def carbon_saved_kwh(
    baseline_kwh: float,
    actual_kwh: float,
    region: str = "default",
) -> dict[str, float]:
    """Compute the carbon savings from reducing energy consumption.

    Args:
        baseline_kwh: Reference energy consumption in kWh (before improvement).
        actual_kwh: Actual energy consumption in kWh (after improvement).
        region: Grid region identifier used for carbon intensity lookup.

    Returns:
        Dict with 'kwh_saved', 'co2_kg_saved', and 'pct_reduction' keys.
        'pct_reduction' is 0.0 when baseline_kwh is zero.
    """
    kwh_saved = baseline_kwh - actual_kwh
    sign = 1.0 if kwh_saved >= 0 else -1.0
    co2_saved = sign * kwh_to_co2_kg(abs(kwh_saved), region)
    pct = (kwh_saved / baseline_kwh * 100.0) if baseline_kwh != 0.0 else 0.0
    return {
        "kwh_saved": round(kwh_saved, 4),
        "co2_kg_saved": round(co2_saved, 4),
        "pct_reduction": round(pct, 4),
    }


def compare_regions(kwh: float, regions: list[str] | None = None) -> list[dict[str, float]]:
    """Compare carbon footprint for *kwh* across multiple grid regions.

    Useful for helping users understand how location choice affects emissions.

    Args:
        kwh: Energy consumption in kilowatt-hours.
        regions: List of region identifiers to compare.  Defaults to all
            known regions in ``GRID_CARBON_INTENSITY``.

    Returns:
        List of dicts with 'region', 'intensity_kg_per_kwh', and 'co2_kg',
        sorted from lowest to highest CO2 impact.

    Raises:
        ValueError: If *kwh* is negative.
    """
    if kwh < 0:
        raise ValueError(f"kwh must be non-negative, got {kwh}")
    if regions is None:
        regions = [r for r in GRID_CARBON_INTENSITY if r != "default"]
    results = [
        {
            "region": r,
            "intensity_kg_per_kwh": _grid_intensity(r),
            "co2_kg": kwh_to_co2_kg(kwh, r),
        }
        for r in regions
    ]
    return sorted(results, key=lambda x: x["co2_kg"])


def carbon_budget_remaining(
    annual_budget_kg: float,
    consumed_kg: float,
) -> dict[str, float]:
    """Compute how much CO2 budget is remaining given consumption so far.

    Args:
        annual_budget_kg: Total annual CO2 allowance in kg.
        consumed_kg: CO2 already emitted in kg.

    Returns:
        Dict with 'remaining_kg', 'used_pct', and 'on_track' (bool as float 1.0/0.0).

    Raises:
        ValueError: If either argument is negative.
    """
    if annual_budget_kg < 0:
        raise ValueError(f"annual_budget_kg must be non-negative, got {annual_budget_kg}")
    if consumed_kg < 0:
        raise ValueError(f"consumed_kg must be non-negative, got {consumed_kg}")
    remaining = max(0.0, annual_budget_kg - consumed_kg)
    used_pct = round(consumed_kg / annual_budget_kg * 100.0, 2) if annual_budget_kg > 0 else 0.0
    on_track = 1.0 if consumed_kg <= annual_budget_kg else 0.0
    return {
        "remaining_kg": round(remaining, 4),
        "used_pct": used_pct,
        "on_track": on_track,
    }
