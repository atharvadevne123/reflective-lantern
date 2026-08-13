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
    actual_kwh: float | None = None,
    baseline_kwh: float | None = None,
    region: str = "default",
    *,
    old_kwh: float | None = None,
    new_kwh: float | None = None,
) -> dict[str, float]:
    """Estimate CO2 savings from reduced energy consumption.

    Args:
        actual_kwh: Measured energy consumption in kWh (or use new_kwh).
        baseline_kwh: Reference baseline consumption in kWh (or use old_kwh).
        region: Grid region identifier.
        old_kwh: Alias for baseline_kwh.
        new_kwh: Alias for actual_kwh.

    Returns:
        Dict with saved_kwh, saved_co2_kg, saved_co2_tonnes, and trees_saved.
    """
    _actual = new_kwh if new_kwh is not None else (actual_kwh or 0.0)
    _baseline = old_kwh if old_kwh is not None else (baseline_kwh or 0.0)
    saved_kwh = max(0.0, _baseline - _actual)
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
    "carbon_per_sqm",
    "carbon_reduction_pct",
    "carbon_saved_kwh",
    "carbon_savings",
    "co2_kg_to_tonnes",
    "compare_regions",
    "daily_carbon_estimate",
    "fleet_emission_factor",
    "kwh_to_co2_kg",
    "lifetime_carbon_savings",
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


def emission_factor_change(
    old_region: str,
    new_region: str,
    kwh: float,
) -> dict[str, float]:
    """Estimate the CO2 impact of switching from one grid region to another.

    Args:
        old_region: Current grid region identifier.
        new_region: Target grid region identifier.
        kwh: Annual energy consumption in kilowatt-hours.

    Returns:
        Dict with old_co2_kg, new_co2_kg, co2_change_kg, and pct_change.

    Raises:
        ValueError: If *kwh* is negative.
    """
    if kwh < 0:
        raise ValueError(f"kwh must be non-negative, got {kwh}")
    old_co2 = kwh_to_co2_kg(kwh, old_region)
    new_co2 = kwh_to_co2_kg(kwh, new_region)
    change = new_co2 - old_co2
    pct = round(change / old_co2 * 100.0, 4) if old_co2 > 0 else 0.0
    return {
        "old_co2_kg": old_co2,
        "new_co2_kg": new_co2,
        "co2_change_kg": round(change, 4),
        "pct_change": pct,
    }


def co2_per_building_type(
    kwh_per_sqft: float,
    sqft: float,
    building_type: str = "office",
    region: str = "default",
) -> dict[str, float]:
    """Estimate CO2 emissions for a building based on type and size.

    Applies a type-based intensity adjustment on top of the regional
    grid intensity, reflecting different usage patterns for building types.

    Args:
        kwh_per_sqft: Annual energy use intensity (kWh per square foot).
        sqft: Floor area in square feet.
        building_type: One of 'office', 'retail', 'industrial', 'residential'.
        region: Grid region identifier.

    Returns:
        Dict with total_kwh, total_co2_kg, and intensity_factor.
    """
    _type_factors: dict[str, float] = {
        "office": 1.0,
        "retail": 1.2,
        "industrial": 1.5,
        "residential": 0.8,
    }
    factor = _type_factors.get(building_type.lower(), 1.0)
    adjusted_kwh = kwh_per_sqft * sqft * factor
    total_co2 = kwh_to_co2_kg(adjusted_kwh, region)
    return {
        "total_kwh": round(adjusted_kwh, 4),
        "total_co2_kg": total_co2,
        "intensity_factor": factor,
    }


def carbon_score(co2_kg: float, max_kg: float) -> float:
    """Compute a normalized 0-100 carbon score (100 = best / cleanest).

    Args:
        co2_kg: Actual CO2 emissions in kilograms.
        max_kg: Theoretical maximum CO2 emissions for this context.

    Returns:
        Score in [0, 100]; 100 when co2_kg == 0, decreasing as emissions rise.
        Clamped to 0 when co2_kg >= max_kg.

    Raises:
        ValueError: If *max_kg* is not positive or *co2_kg* is negative.
    """
    if max_kg <= 0:
        raise ValueError(f"max_kg must be positive, got {max_kg}")
    if co2_kg < 0:
        raise ValueError(f"co2_kg must be non-negative, got {co2_kg}")
    score = max(0.0, (1.0 - co2_kg / max_kg) * 100.0)
    return round(score, 4)


def annual_emission_estimate(monthly_kwh: list[float], emission_factor: float) -> float:
    """Estimate total annual CO2 emissions from monthly consumption data.

    Args:
        monthly_kwh: List of monthly energy consumption values in kWh.
        emission_factor: kg CO2 per kWh.

    Returns:
        Estimated annual CO2 in kg.

    Raises:
        ValueError: If monthly_kwh is empty or emission_factor is non-positive.
    """
    if not monthly_kwh:
        raise ValueError("monthly_kwh must be non-empty")
    if emission_factor <= 0:
        raise ValueError("emission_factor must be positive")
    total_kwh = sum(monthly_kwh)
    if len(monthly_kwh) < 12:
        total_kwh = total_kwh / len(monthly_kwh) * 12
    return round(total_kwh * emission_factor, 4)


def carbon_reduction_potential(
    baseline_kwh: float,
    target_kwh: float,
    emission_factor: float,
) -> float:
    """Compute CO2 reduction potential when reducing consumption from baseline to target.

    Args:
        baseline_kwh: Current energy consumption in kWh.
        target_kwh: Target energy consumption in kWh.
        emission_factor: kg CO2 per kWh.

    Returns:
        CO2 reduction in kg (positive = reduction achieved).

    Raises:
        ValueError: If emission_factor is non-positive.
    """
    if emission_factor <= 0:
        raise ValueError("emission_factor must be positive")
    return round((baseline_kwh - target_kwh) * emission_factor, 4)


def lifetime_carbon_savings(
    annual_kwh_saved: float,
    lifetime_years: int,
    region: str = "default",
) -> dict[str, float]:
    """Compute total carbon savings over a system's operating lifetime.

    Args:
        annual_kwh_saved: kWh saved per year by the efficiency improvement.
        lifetime_years: Expected operational lifetime in years.
        region: Grid region identifier for carbon intensity lookup.

    Returns:
        Dict with 'total_kwh_saved', 'total_co2_kg_saved', 'total_co2_tonnes_saved'.
    """
    total_kwh = annual_kwh_saved * lifetime_years
    total_co2_kg = kwh_to_co2_kg(total_kwh, region)
    return {
        "total_kwh_saved": round(total_kwh, 4),
        "total_co2_kg_saved": round(total_co2_kg, 4),
        "total_co2_tonnes_saved": co2_kg_to_tonnes(total_co2_kg),
    }


def fleet_emission_factor(
    vehicles: list[dict],
    kwh_key: str = "annual_kwh",
    region_key: str = "region",
    default_region: str = "default",
) -> dict[str, float]:
    """Compute aggregate CO2 emissions for a fleet of energy consumers.

    Each vehicle/device is a dict with optional ``annual_kwh`` and ``region`` keys.

    Args:
        vehicles: List of dicts representing fleet assets.
        kwh_key: Key in each dict for annual kWh consumption.
        region_key: Key in each dict for the grid region.
        default_region: Fallback region when ``region_key`` is absent.

    Returns:
        Dict with 'total_kwh', 'total_co2_kg', 'mean_co2_kg_per_asset'.
    """
    total_kwh = 0.0
    total_co2 = 0.0
    count = 0
    for v in vehicles:
        kwh = float(v.get(kwh_key, 0.0))
        region = str(v.get(region_key, default_region))
        co2 = kwh_to_co2_kg(kwh, region)
        total_kwh += kwh
        total_co2 += co2
        count += 1
    mean_co2 = total_co2 / count if count else 0.0
    return {
        "total_kwh": round(total_kwh, 4),
        "total_co2_kg": round(total_co2, 4),
        "mean_co2_kg_per_asset": round(mean_co2, 4),
    }


def carbon_per_sqm(
    annual_kwh: float,
    floor_area_sqm: float,
    region: str = "default",
) -> float:
    """Compute building carbon intensity in kg CO2e per square metre per year.

    Args:
        annual_kwh: Total annual energy consumption in kWh.
        floor_area_sqm: Building floor area in square metres (must be > 0).
        region: Grid region identifier for the carbon intensity factor.

    Returns:
        Carbon intensity in kg CO2e/m²/year, rounded to 4 decimal places.

    Raises:
        ValueError: If *floor_area_sqm* is non-positive.
    """
    if floor_area_sqm <= 0:
        raise ValueError(f"floor_area_sqm must be positive, got {floor_area_sqm}")
    total_co2 = kwh_to_co2_kg(annual_kwh, region)
    return round(total_co2 / floor_area_sqm, 4)


def carbon_reduction_pct(
    baseline_co2_kg: float,
    actual_co2_kg: float,
) -> float:
    """Compute the percentage reduction in CO2 emissions vs a baseline.

    Args:
        baseline_co2_kg: Reference (pre-intervention) CO2 in kg.
        actual_co2_kg: Post-intervention CO2 in kg.

    Returns:
        Percentage reduction; positive means emissions went down.
        Returns 0.0 when *baseline_co2_kg* is zero.
    """
    if baseline_co2_kg == 0.0:
        return 0.0
    return round((baseline_co2_kg - actual_co2_kg) / baseline_co2_kg * 100.0, 4)


def cumulative_co2(daily_kg: list[float]) -> list[float]:
    """Return the running cumulative CO2 total from a series of daily values.

    Args:
        daily_kg: Daily CO2 emissions in kilograms.

    Returns:
        Cumulative CO2 list of the same length; entry i = sum(daily_kg[0:i+1]).
    """
    total = 0.0
    result: list[float] = []
    for v in daily_kg:
        total += v
        result.append(round(total, 6))
    return result


def carbon_per_occupant(co2_kg: float, occupants: int) -> float:
    """Return per-occupant CO2 footprint.

    Args:
        co2_kg: Total CO2 in kilograms for the period.
        occupants: Number of occupants (must be > 0).

    Returns:
        Per-occupant share in kg, rounded to 4 decimal places.

    Raises:
        ValueError: If *occupants* is not positive.
    """
    if occupants <= 0:
        raise ValueError(f"occupants must be positive, got {occupants}")
    return round(co2_kg / occupants, 4)


def net_zero_timeline(
    annual_co2_kg: float,
    annual_reduction_pct: float,
    offset_co2_kg_per_year: float = 0.0,
    max_years: int = 100,
) -> dict[str, object]:
    """Estimate years to reach net-zero given a yearly reduction rate and offsets.

    Args:
        annual_co2_kg: Starting annual CO2 emissions in kg.
        annual_reduction_pct: Percentage reduction applied each year (0-100).
        offset_co2_kg_per_year: CO2 offset added each year (e.g. renewable credits).
        max_years: Maximum number of years to simulate before giving up.

    Returns:
        Dict with ``years_to_net_zero`` (int or None if not reached),
        ``trajectory`` (list of annual totals), and ``achieved`` (bool).

    Raises:
        ValueError: If *annual_reduction_pct* is outside [0, 100).
    """
    if not (0.0 <= annual_reduction_pct < 100.0):
        raise ValueError("annual_reduction_pct must be in [0, 100)")

    rate = annual_reduction_pct / 100.0
    current = float(annual_co2_kg)
    trajectory: list[float] = []
    years_to_net_zero: int | None = None

    for year in range(1, max_years + 1):
        current = max(0.0, current * (1.0 - rate) - offset_co2_kg_per_year)
        trajectory.append(round(current, 2))
        if current <= 0.0 and years_to_net_zero is None:
            years_to_net_zero = year
            break

    achieved = years_to_net_zero is not None
    return {
        "years_to_net_zero": years_to_net_zero,
        "trajectory": trajectory,
        "achieved": achieved,
    }
