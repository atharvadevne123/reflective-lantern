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
    "list_building_types",
    "site_eui",
]
