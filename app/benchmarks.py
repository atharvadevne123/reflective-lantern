"""Building energy benchmarking utilities: Energy Use Intensity (EUI) calculations."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ASHRAE reference EUI values (kWh/m² per year) by building type
ASHRAE_EUI_BENCHMARKS: dict[str, float] = {
    "office": 176.0,
    "retail": 220.0,
    "school": 130.0,
    "hospital": 580.0,
    "hotel": 280.0,
    "warehouse": 90.0,
    "apartment": 120.0,
    "restaurant": 750.0,
    "data_center": 1200.0,
    "manufacturing": 400.0,
    "default": 200.0,
}


def compute_eui(annual_kwh: float, floor_area_sqm: float) -> float:
    """Compute Energy Use Intensity (EUI) in kWh/m² per year.

    Args:
        annual_kwh: Total annual energy consumption in kWh.
        floor_area_sqm: Building floor area in square metres.

    Returns:
        EUI as kWh/m²/year, rounded to 2 decimal places.

    Raises:
        ValueError: If *floor_area_sqm* is non-positive or *annual_kwh* is negative.
    """
    if floor_area_sqm <= 0:
        raise ValueError(f"floor_area_sqm must be positive, got {floor_area_sqm}")
    if annual_kwh < 0:
        raise ValueError(f"annual_kwh must be non-negative, got {annual_kwh}")
    return round(annual_kwh / floor_area_sqm, 2)


def benchmark_eui(
    eui: float,
    building_type: str = "default",
) -> dict[str, object]:
    """Compare an EUI against the ASHRAE benchmark for the given building type.

    Args:
        eui: Computed Energy Use Intensity in kWh/m²/year.
        building_type: Building type key from ASHRAE_EUI_BENCHMARKS.

    Returns:
        Dict with 'benchmark_eui', 'ratio', 'rating' ('excellent'|'good'|'average'|'poor'),
        and 'building_type'.
    """
    benchmark = ASHRAE_EUI_BENCHMARKS.get(building_type.lower(), ASHRAE_EUI_BENCHMARKS["default"])
    ratio = round(eui / benchmark, 4) if benchmark > 0 else float("inf")

    if ratio <= 0.7:
        rating = "excellent"
    elif ratio <= 0.9:
        rating = "good"
    elif ratio <= 1.1:
        rating = "average"
    else:
        rating = "poor"

    logger.info(
        "EUI benchmark: type=%s eui=%.2f benchmark=%.2f ratio=%.3f rating=%s",
        building_type,
        eui,
        benchmark,
        ratio,
        rating,
    )
    return {
        "building_type": building_type,
        "eui": eui,
        "benchmark_eui": benchmark,
        "ratio": ratio,
        "rating": rating,
    }


def annual_to_monthly_estimate(annual_kwh: float, profile: list[float] | None = None) -> list[float]:
    """Distribute annual kWh across 12 months using an optional seasonal profile.

    Args:
        annual_kwh: Total annual energy consumption in kWh.
        profile: 12-element list of relative weights per month. If None,
            a default slightly winter-heavy profile is used.

    Returns:
        12-element list of estimated monthly kWh values.

    Raises:
        ValueError: If *profile* is provided but does not have exactly 12 elements.
    """
    if profile is not None and len(profile) != 12:
        raise ValueError(f"profile must have exactly 12 elements, got {len(profile)}")
    if profile is None:
        profile = [1.1, 1.0, 0.95, 0.85, 0.85, 0.90, 1.05, 1.05, 0.90, 0.85, 0.95, 1.10]
    total_weight = sum(profile)
    return [round(annual_kwh * w / total_weight, 2) for w in profile]


def list_building_types() -> list[str]:
    """Return sorted list of supported building type keys."""
    return sorted(ASHRAE_EUI_BENCHMARKS.keys())


def site_eui(annual_kwh: float, floor_area_sqft: float) -> float:
    """Compute Site Energy Use Intensity from annual kWh and floor area in sq ft.

    Converts square feet to square metres internally before computing EUI.

    Args:
        annual_kwh: Total annual energy consumption in kWh.
        floor_area_sqft: Building floor area in square feet.

    Returns:
        EUI in kWh/m²/year, rounded to 2 decimal places.

    Raises:
        ValueError: If *floor_area_sqft* is non-positive or *annual_kwh* is negative.
    """
    if floor_area_sqft <= 0:
        raise ValueError(f"floor_area_sqft must be positive, got {floor_area_sqft}")
    sqm = floor_area_sqft * 0.092903
    return compute_eui(annual_kwh, sqm)


def compare_buildings(
    buildings: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Rank buildings by EUI and append benchmark ratings.

    Each building dict must contain 'name', 'annual_kwh', 'floor_area_sqm',
    and optionally 'building_type'.

    Args:
        buildings: List of building descriptor dicts.

    Returns:
        List of result dicts with EUI, benchmark, rating, and rank (1 = best).
    """
    results = []
    for b in buildings:
        name = str(b.get("name", "unknown"))
        kwh = float(b.get("annual_kwh", 0))
        sqm = float(b.get("floor_area_sqm", 1))
        btype = str(b.get("building_type", "default"))
        eui_val = compute_eui(kwh, sqm)
        bmark = benchmark_eui(eui_val, btype)
        results.append({"name": name, **bmark})
    results.sort(key=lambda r: float(r["eui"]))
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank
    return results


__all__ = [
    "annual_to_monthly_estimate",
    "benchmark_eui",
    "compare_buildings",
    "compute_eui",
    "efficiency_gap",
    "energy_intensity_ratio",
    "eui_percentile_category",
    "list_building_types",
    "site_eui",
    "target_eui",
]


def energy_intensity_ratio(
    actual_eui: float,
    reference_eui: float,
) -> float:
    """Compute the Energy Intensity Ratio (EIR) = actual EUI / reference EUI.

    A ratio < 1.0 means better than reference; > 1.0 means worse.

    Args:
        actual_eui: Computed EUI for the building (kWh/m²/year).
        reference_eui: Reference or target EUI for comparison.

    Returns:
        EIR rounded to 4 decimal places, or 0.0 if reference_eui is zero.
    """
    if reference_eui == 0.0:
        return 0.0
    return round(actual_eui / reference_eui, 4)


def efficiency_gap(
    actual_kwh: float,
    target_kwh: float,
) -> float:
    """Compute the efficiency gap as the percentage reduction needed to hit target.

    A positive result means actual exceeds target (needs improvement);
    negative means actual is already below target (exceeds goal).

    Args:
        actual_kwh: Measured energy consumption in kWh.
        target_kwh: Target energy consumption in kWh.

    Returns:
        Gap as a percentage of target, rounded to 4 decimal places.
        Returns 0.0 if target_kwh is zero.
    """
    if target_kwh == 0.0:
        return 0.0
    return round((actual_kwh - target_kwh) / target_kwh * 100.0, 4)


def carbon_intensity_benchmark(kwh: float, emission_factor: float, floor_area_sqm: float) -> float:
    """Calculate carbon intensity benchmark (kg CO2 per sqm).

    Args:
        kwh: Total energy consumed in kWh.
        emission_factor: kg CO2 per kWh.
        floor_area_sqm: Floor area in square metres.

    Returns:
        Carbon intensity in kg CO2 per sqm.

    Raises:
        ValueError: If any argument is non-positive.
    """
    if kwh <= 0 or emission_factor <= 0 or floor_area_sqm <= 0:
        raise ValueError("All arguments must be positive")
    return round(kwh * emission_factor / floor_area_sqm, 4)


def percentage_below_benchmark(actual_eui: float, benchmark_eui: float) -> float:
    """Return percentage difference of actual EUI below benchmark (negative = over benchmark).

    Args:
        actual_eui: Measured EUI value.
        benchmark_eui: Target benchmark EUI value.

    Returns:
        Percentage below benchmark (positive = better than benchmark).

    Raises:
        ValueError: If benchmark_eui is zero.
    """
    if benchmark_eui == 0:
        raise ValueError("benchmark_eui cannot be zero")
    return round((benchmark_eui - actual_eui) / benchmark_eui * 100.0, 4)


def weighted_average_eui(euis: list[float], weights: list[float]) -> float:
    """Compute weighted average EUI across multiple buildings.

    Args:
        euis: List of EUI values.
        weights: Corresponding weights (e.g., floor areas).

    Returns:
        Weighted average EUI.

    Raises:
        ValueError: If lists are empty or of different lengths.
    """
    if not euis or not weights:
        raise ValueError("euis and weights must be non-empty")
    if len(euis) != len(weights):
        raise ValueError("euis and weights must have the same length")
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Total weight cannot be zero")
    return round(sum(e * w for e, w in zip(euis, weights, strict=False)) / total_weight, 4)


def energy_star_score(eui: float, median_eui: float, best_eui: float) -> float:
    """Estimate an ENERGY STAR-like score (0-100) for a building.

    Args:
        eui: Building's EUI.
        median_eui: Median EUI for the building type.
        best_eui: Best (lowest) achievable EUI for the type.

    Returns:
        Score between 0 and 100.

    Raises:
        ValueError: If median_eui equals best_eui.
    """
    if median_eui == best_eui:
        raise ValueError("median_eui and best_eui cannot be equal")
    score = 50 + 50 * (median_eui - eui) / (median_eui - best_eui)
    return round(max(0.0, min(100.0, score)), 2)


def target_eui(building_type: str, improvement_pct: float = 20.0) -> float:
    """Return a target EUI for *building_type* given a percentage improvement goal.

    Args:
        building_type: Building type key in ASHRAE_EUI_BENCHMARKS.
        improvement_pct: Desired percentage improvement over the benchmark (0-100).

    Returns:
        Target EUI in kWh/m²/year, rounded to 2 decimal places.

    Raises:
        ValueError: If *improvement_pct* is not in [0, 100].
    """
    if not (0.0 <= improvement_pct <= 100.0):
        raise ValueError(f"improvement_pct must be in [0, 100], got {improvement_pct}")
    benchmark = ASHRAE_EUI_BENCHMARKS.get(building_type.lower(), ASHRAE_EUI_BENCHMARKS["default"])
    return round(benchmark * (1.0 - improvement_pct / 100.0), 2)


def eui_percentile_category(eui: float, building_type: str = "default") -> str:
    """Classify an EUI into an ENERGY STAR-style percentile category.

    Categories relative to the ASHRAE benchmark for *building_type*:
    - 'top_25'  : EUI <= 50% of benchmark (very efficient)
    - 'median'  : EUI <= 90% of benchmark
    - 'average' : EUI <= 110% of benchmark
    - 'bottom_25': EUI > 110% of benchmark (inefficient)

    Args:
        eui: Building EUI in kWh/m²/year.
        building_type: Building type for benchmark lookup.

    Returns:
        Category string: 'top_25', 'median', 'average', or 'bottom_25'.
    """
    benchmark = ASHRAE_EUI_BENCHMARKS.get(building_type.lower(), ASHRAE_EUI_BENCHMARKS["default"])
    ratio = eui / benchmark if benchmark > 0 else float("inf")
    if ratio <= 0.5:
        return "top_25"
    if ratio <= 0.9:
        return "median"
    if ratio <= 1.1:
        return "average"
    return "bottom_25"
