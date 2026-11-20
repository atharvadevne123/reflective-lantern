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
    "annual_carbon_budget",
    "annual_carbon_report",
    "annual_carbon_trajectory",
    "carbon_budget_remaining",
    "carbon_budget_status",
    "carbon_intensity_by_hour",
    "carbon_intensity_category",
    "carbon_intensity_label",
    "carbon_intensity_rank",
    "carbon_offset_cost",
    "carbon_per_sqm",
    "carbon_reduction_pct",
    "carbon_saved_kwh",
    "carbon_savings",
    "carbon_savings_vs_baseline",
    "co2_kg_to_tonnes",
    "compare_regions",
    "daily_carbon_estimate",
    "fleet_emission_factor",
    "hourly_carbon_profile",
    "kwh_to_co2_kg",
    "lifetime_carbon_savings",
    "monthly_co2_breakdown",
    "renewable_offset_factor",
    "tree_offset_days",
    "trees_equivalent",
    "weighted_carbon_factor",
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


def carbon_intensity_label(intensity_kg_per_kwh: float) -> str:
    """Return a descriptive label for a carbon intensity value.

    Args:
        intensity_kg_per_kwh: Grid carbon intensity in kg CO2e per kWh.

    Returns:
        One of 'very_low', 'low', 'moderate', 'high', 'very_high'.
    """
    if intensity_kg_per_kwh < 0.15:
        return "very_low"
    if intensity_kg_per_kwh < 0.25:
        return "low"
    if intensity_kg_per_kwh < 0.40:
        return "medium"
    if intensity_kg_per_kwh < 0.55:
        return "high"
    return "very_high"


def annual_carbon_budget(
    target_co2_tonnes: float,
    current_co2_kg: float,
    year_fraction_elapsed: float = 0.0,
) -> dict[str, float]:
    """Compute carbon budget remaining for the year given a tonne target.

    Args:
        target_co2_tonnes: Annual CO2 budget in tonnes.
        current_co2_kg: CO2 already emitted this year in kilograms.
        year_fraction_elapsed: Fraction of the year elapsed (0.0 to 1.0).

    Returns:
        Dict with 'budget_kg', 'spent_kg', 'remaining_kg',
        'on_track' (bool), and 'projected_annual_kg'.

    Raises:
        ValueError: If *year_fraction_elapsed* is outside [0, 1].
    """
    if not (0.0 <= year_fraction_elapsed <= 1.0):
        raise ValueError("year_fraction_elapsed must be between 0 and 1")
    budget_kg = target_co2_tonnes * 1000.0
    remaining_kg = max(0.0, budget_kg - current_co2_kg)
    projected = (current_co2_kg / year_fraction_elapsed) if year_fraction_elapsed > 0 else current_co2_kg
    on_track = projected <= budget_kg
    return {
        "budget_kg": round(budget_kg, 2),
        "spent_kg": round(current_co2_kg, 2),
        "remaining_kg": round(remaining_kg, 2),
        "on_track": on_track,
        "projected_annual_kg": round(projected, 2),
    }


def carbon_budget_status(budget_kg: float, consumed_kg: float) -> dict[str, float]:
    """Compute a snapshot of carbon-budget usage.

    Args:
        budget_kg: Total carbon budget in kilograms of CO₂ (must be positive).
        consumed_kg: Amount already emitted, in kilograms (must be non-negative).

    Returns:
        Dict with ``budget_kg``, ``consumed_kg``, ``consumed_pct``,
        ``remaining_kg`` (negative when over-budget), and ``overage_kg``.

    Raises:
        ValueError: If ``budget_kg`` is not strictly positive or
            ``consumed_kg`` is negative.
    """
    if budget_kg <= 0:
        raise ValueError(f"budget_kg must be positive, got {budget_kg}")
    if consumed_kg < 0:
        raise ValueError(f"consumed_kg must be non-negative, got {consumed_kg}")
    remaining = budget_kg - consumed_kg
    overage = max(0.0, -remaining)
    consumed_pct = (consumed_kg / budget_kg) * 100.0
    return {
        "budget_kg": round(budget_kg, 4),
        "consumed_kg": round(consumed_kg, 4),
        "consumed_pct": round(consumed_pct, 4),
        "remaining_kg": round(remaining, 4),
        "overage_kg": round(overage, 4),
    }


def weighted_carbon_factor(sources: list[dict[str, float]]) -> float:
    """Compute the emissions-weighted mean carbon factor across a fuel mix.

    Each source dict must have ``fraction`` (sums to 1) and ``factor_kg_per_kwh``.

    Args:
        sources: List of dicts describing the energy mix.

    Returns:
        Weighted mean factor in kg CO₂ per kWh.

    Raises:
        ValueError: If *sources* is empty or fractions don't sum to (approximately) 1.
    """
    if not sources:
        raise ValueError("sources must not be empty")
    total_fraction = sum(s.get("fraction", 0.0) for s in sources)
    if abs(total_fraction - 1.0) > 0.05:
        raise ValueError(f"source fractions must sum to 1, got {total_fraction}")
    weighted = sum(s.get("fraction", 0.0) * s.get("factor_kg_per_kwh", 0.0) for s in sources)
    return round(weighted, 6)


def carbon_intensity_category(factor_kg_per_kwh: float) -> str:
    """Classify a grid carbon factor into a qualitative bucket.

    Buckets:
        * ``factor < 0.1`` → ``"very_low"``
        * ``factor < 0.3`` → ``"low"``
        * ``factor < 0.5`` → ``"medium"``
        * otherwise → ``"high"``

    Args:
        factor_kg_per_kwh: Emission factor (kg CO₂ per kWh); must be non-negative.

    Returns:
        Category label.

    Raises:
        ValueError: If ``factor_kg_per_kwh`` is negative.
    """
    if factor_kg_per_kwh < 0:
        raise ValueError(f"factor must be non-negative, got {factor_kg_per_kwh}")
    if factor_kg_per_kwh < 0.1:
        return "very_low"
    if factor_kg_per_kwh < 0.3:
        return "low"
    if factor_kg_per_kwh < 0.5:
        return "medium"
    return "high"


def carbon_offset_cost(co2_kg: float, cost_per_tonne: float = 15.0) -> float:
    """Estimate the market cost of offsetting *co2_kg* of emissions.

    Args:
        co2_kg: Kilograms of CO₂ to offset; must be non-negative.
        cost_per_tonne: Offset price in USD per metric tonne (default $15).

    Returns:
        Cost in USD.

    Raises:
        ValueError: If ``co2_kg`` is negative or ``cost_per_tonne`` is negative.
    """
    if co2_kg < 0:
        raise ValueError(f"co2_kg must be non-negative, got {co2_kg}")
    if cost_per_tonne < 0:
        raise ValueError(f"cost_per_tonne must be non-negative, got {cost_per_tonne}")
    tonnes = co2_kg / 1000.0
    return round(tonnes * cost_per_tonne, 4)


def annual_carbon_trajectory(monthly_kg: list[float]) -> dict[str, float]:
    """Summarise a series of monthly CO₂ totals into an annual trajectory.

    Args:
        monthly_kg: Sequence of monthly emission totals (kg). Must be non-empty
            and non-negative.

    Returns:
        Dict with ``total_kg``, ``monthly_avg_kg``, ``peak_month_kg``, and
        ``trend_slope`` (least-squares slope of monthly totals).

    Raises:
        ValueError: If ``monthly_kg`` is empty or contains negative values.
    """
    if not monthly_kg:
        raise ValueError("monthly_kg must not be empty")
    if any(v < 0 for v in monthly_kg):
        raise ValueError("monthly_kg values must be non-negative")
    n = len(monthly_kg)
    total = float(sum(monthly_kg))
    avg = total / n
    peak = float(max(monthly_kg))
    if n < 2:
        slope = 0.0
    else:
        mean_x = (n - 1) / 2.0
        mean_y = avg
        num = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(monthly_kg))
        den = sum((i - mean_x) ** 2 for i in range(n))
        slope = num / den if den > 0 else 0.0
    return {
        "total_kg": round(total, 4),
        "monthly_avg_kg": round(avg, 4),
        "peak_month_kg": round(peak, 4),
        "trend_slope": round(slope, 6),
    }


def carbon_savings_vs_baseline(actual_kg: float, baseline_kg: float) -> dict[str, float]:
    """Compute absolute and percentage carbon savings against a baseline.

    Args:
        actual_kg: Actual emissions in kg (must be non-negative).
        baseline_kg: Baseline (typical or prior) emissions in kg (must be non-negative).

    Returns:
        Dict with ``savings_kg`` (baseline - actual; negative when worse) and
        ``savings_pct`` (percentage; 0 when baseline is zero).

    Raises:
        ValueError: If either value is negative.
    """
    if actual_kg < 0:
        raise ValueError(f"actual_kg must be non-negative, got {actual_kg}")
    if baseline_kg < 0:
        raise ValueError(f"baseline_kg must be non-negative, got {baseline_kg}")
    savings = baseline_kg - actual_kg
    pct = (savings / baseline_kg) * 100.0 if baseline_kg > 0 else 0.0
    return {
        "savings_kg": round(savings, 4),
        "savings_pct": round(pct, 4),
    }


def carbon_per_capita(total_kg: float, population: int) -> float:
    """Return per-capita carbon emissions (kg CO₂ per person).

    Args:
        total_kg: Total emissions in kilograms (must be non-negative).
        population: Number of people (must be strictly positive).

    Returns:
        Emissions per person in kg.

    Raises:
        ValueError: If ``total_kg`` is negative or ``population`` is non-positive.
    """
    if total_kg < 0:
        raise ValueError(f"total_kg must be non-negative, got {total_kg}")
    if population <= 0:
        raise ValueError(f"population must be positive, got {population}")
    return round(total_kg / population, 4)


def carbon_reduction_years(annual_kg: float, reduction_pct: float, target_kg: float) -> float:
    """Estimate the number of years to hit *target_kg* by cutting *reduction_pct* each year.

    Args:
        annual_kg: Current annual emissions (must be strictly positive).
        reduction_pct: Annual reduction as a percentage in (0, 100].
        target_kg: Target annual emissions (must be non-negative).

    Returns:
        Number of years (float). Returns 0.0 when already at or below target,
        or ``inf`` when the reduction never gets there.

    Raises:
        ValueError: If arguments are outside expected ranges.
    """
    if annual_kg <= 0:
        raise ValueError(f"annual_kg must be positive, got {annual_kg}")
    if not 0 < reduction_pct <= 100:
        raise ValueError(f"reduction_pct must be in (0, 100], got {reduction_pct}")
    if target_kg < 0:
        raise ValueError(f"target_kg must be non-negative, got {target_kg}")
    if annual_kg <= target_kg:
        return 0.0
    if target_kg == 0:
        return float("inf")
    import math as _math

    factor = 1.0 - reduction_pct / 100.0
    if factor <= 0:
        return 1.0
    return round(_math.log(target_kg / annual_kg) / _math.log(factor), 4)


def emissions_reduction_pct(baseline_kg: float, current_kg: float) -> float:
    """Return the percentage reduction in emissions relative to a baseline.

    Args:
        baseline_kg: Reference (historical) emissions in kg CO2.
        current_kg: Current emissions in kg CO2.

    Returns:
        Positive value when emissions fell, negative when they rose. 0.0 when baseline is zero.
    """
    if baseline_kg == 0.0:
        return 0.0
    return round((baseline_kg - current_kg) / baseline_kg * 100.0, 4)


def carbon_intensity_trend(
    intensities: list[float],
) -> dict[str, object]:
    """Analyse a sequence of carbon intensity readings for trend direction.

    Args:
        intensities: Ordered carbon intensity values (kg CO2/kWh).

    Returns:
        Dict with 'mean', 'min', 'max', 'trend' ('improving'|'worsening'|'stable'),
        and 'latest_label'. Returns a zeros dict when *intensities* is empty.
    """
    if not intensities:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "trend": "stable", "latest_label": "very_low"}
    mean_val = sum(intensities) / len(intensities)
    if len(intensities) >= 2:
        delta = intensities[-1] - intensities[0]
        trend = "worsening" if delta > 0.01 else ("improving" if delta < -0.01 else "stable")
    else:
        trend = "stable"
    return {
        "mean": round(mean_val, 6),
        "min": round(min(intensities), 6),
        "max": round(max(intensities), 6),
        "trend": trend,
        "latest_label": carbon_intensity_label(intensities[-1]),
    }


def renewable_offset_factor(renewable_pct: float) -> float:
    """Return the effective carbon intensity multiplier after accounting for renewables.

    Args:
        renewable_pct: Percentage of energy from renewable sources (0-100).

    Returns:
        Multiplier in range [0.0, 1.0]; 0.0 means fully renewable.

    Raises:
        ValueError: If renewable_pct is outside [0, 100].
    """
    if not (0.0 <= renewable_pct <= 100.0):
        raise ValueError(f"renewable_pct must be in [0, 100], got {renewable_pct}")
    return round(1.0 - renewable_pct / 100.0, 6)


def hourly_carbon_profile(
    hourly_kwh: list[float],
    region: str = "default",
    *,
    renewable_pct: float = 0.0,
) -> list[float]:
    """Compute per-hour CO2 emissions (kg) for a 24-hour energy consumption profile.

    Args:
        hourly_kwh: Energy readings for each hour (must have exactly 24 values).
        region: Grid region identifier for intensity lookup.
        renewable_pct: Percentage of renewables that reduces effective intensity.

    Returns:
        List of 24 CO2-kg values, one per hour.

    Raises:
        ValueError: If hourly_kwh does not contain exactly 24 values.
    """
    if len(hourly_kwh) != 24:
        raise ValueError(f"hourly_kwh must have 24 values, got {len(hourly_kwh)}")
    factor = renewable_offset_factor(renewable_pct)
    intensity = _grid_intensity(region.lower())
    return [round(kwh * intensity * factor, 6) for kwh in hourly_kwh]


def carbon_intensity_rank(region: str) -> int:
    """Rank *region* by grid carbon intensity (1 = cleanest, higher = dirtier).

    Args:
        region: Grid region identifier (case-insensitive).

    Returns:
        Integer rank where 1 is the cleanest region.
    """
    sorted_regions = sorted(GRID_CARBON_INTENSITY.items(), key=lambda kv: kv[1])
    intensity = _grid_intensity(region.lower())
    for rank, (_, val) in enumerate(sorted_regions, start=1):
        if abs(val - intensity) < 1e-9:
            return rank
    return len(sorted_regions)


def cumulative_budget_usage(consumed_co2_kg: float, budget_co2_kg: float) -> dict[str, float]:
    """Compute cumulative carbon budget usage statistics.

    Args:
        consumed_co2_kg: CO2 consumed so far (must be non-negative).
        budget_co2_kg: Total CO2 budget (must be positive).

    Returns:
        Dict with 'used_pct', 'remaining_kg', and 'remaining_pct' keys.

    Raises:
        ValueError: If *budget_co2_kg* is not positive or *consumed_co2_kg* is negative.
    """
    if budget_co2_kg <= 0:
        raise ValueError(f"budget_co2_kg must be positive, got {budget_co2_kg}")
    if consumed_co2_kg < 0:
        raise ValueError(f"consumed_co2_kg must be non-negative, got {consumed_co2_kg}")
    used_pct = min(consumed_co2_kg / budget_co2_kg * 100.0, 100.0)
    remaining_kg = max(budget_co2_kg - consumed_co2_kg, 0.0)
    remaining_pct = max(100.0 - used_pct, 0.0)
    return {
        "used_pct": round(used_pct, 4),
        "remaining_kg": round(remaining_kg, 4),
        "remaining_pct": round(remaining_pct, 4),
    }


def grid_emission_factor(region: str) -> float:
    """Return the CO2 emission factor (kg/kWh) for *region*.

    Args:
        region: Region key (same as those recognised by kwh_to_co2_kg).

    Returns:
        Emission factor in kg CO2 per kWh; falls back to the default factor
        when the region is unknown.
    """
    factors: dict[str, float] = {
        "default": 0.0001,
        "northeast": 0.00014,
        "midwest": 0.00045,
        "south": 0.00042,
        "west": 0.00028,
        "pacific_nw": 0.00010,
        "new_england": 0.00016,
        "texas": 0.00038,
        "florida": 0.00040,
    }
    return factors.get(region.lower(), factors["default"])


def annual_co2_savings(baseline_kwh: float, improved_kwh: float, region: str = "default") -> float:
    """Compute the annual CO2 savings (kg) from an energy-efficiency improvement.

    Args:
        baseline_kwh: Annual consumption before improvement (must be non-negative).
        improved_kwh: Annual consumption after improvement (must be non-negative).
        region: Region key for emission factor lookup.

    Returns:
        CO2 savings in kg; 0.0 when improved_kwh >= baseline_kwh.

    Raises:
        ValueError: If either consumption value is negative.
    """
    if baseline_kwh < 0:
        raise ValueError(f"baseline_kwh must be non-negative, got {baseline_kwh}")
    if improved_kwh < 0:
        raise ValueError(f"improved_kwh must be non-negative, got {improved_kwh}")
    saved_kwh = max(0.0, baseline_kwh - improved_kwh)
    factor = grid_emission_factor(region)
    return round(saved_kwh * factor * 1000.0, 4)
