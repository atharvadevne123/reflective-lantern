"""Energy intensity benchmarking and sector comparison utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SECTOR_BENCHMARKS_KWH_PER_M2: dict[str, float] = {
    "office": 180.0,
    "retail": 320.0,
    "hospital": 650.0,
    "school": 120.0,
    "hotel": 300.0,
    "warehouse": 75.0,
    "data_center": 1200.0,
    "restaurant": 700.0,
    "supermarket": 550.0,
    "residential": 100.0,
}
"""Typical energy use intensity (EUI) benchmarks in kWh/m²/year by sector."""

ENERGY_STAR_THRESHOLD_PERCENTILE: float = 75.0
"""ENERGY STAR certification requires performance at or above 75th percentile."""


@dataclass
class EUIResult:
    """Energy use intensity analysis result."""

    annual_kwh: float
    floor_area_m2: float
    eui_kwh_per_m2: float
    sector: str
    sector_benchmark_kwh_per_m2: float
    performance_ratio: float
    percent_above_benchmark: float
    rating: str


@dataclass
class SectorRanking:
    """Ranking of a building within its sector peer group."""

    asset_id: str
    eui_kwh_per_m2: float
    sector: str
    percentile: float
    rank: int
    total_peers: int
    energy_star_eligible: bool


def energy_use_intensity(
    annual_kwh: float,
    floor_area_m2: float,
) -> float:
    """Compute Energy Use Intensity (EUI).

    Args:
        annual_kwh: Total annual energy consumption in kWh.
        floor_area_m2: Gross floor area in m².

    Returns:
        EUI in kWh/m²/year.

    Raises:
        ValueError: If annual_kwh < 0 or floor_area_m2 <= 0.
    """
    if annual_kwh < 0:
        raise ValueError("annual_kwh must be non-negative")
    if floor_area_m2 <= 0:
        raise ValueError("floor_area_m2 must be positive")
    return round(annual_kwh / floor_area_m2, 4)


def normalise_weather(
    eui_kwh_per_m2: float,
    actual_hdd: float,
    baseline_hdd: float,
) -> float:
    """Weather-normalise an EUI using heating degree days.

    Args:
        eui_kwh_per_m2: Raw EUI in kWh/m²/year.
        actual_hdd: Actual heating degree days in the measurement period.
        baseline_hdd: Long-run baseline heating degree days.

    Returns:
        Weather-normalised EUI in kWh/m²/year.

    Raises:
        ValueError: If any value is non-positive.
    """
    if eui_kwh_per_m2 <= 0:
        raise ValueError("eui_kwh_per_m2 must be positive")
    if actual_hdd <= 0:
        raise ValueError("actual_hdd must be positive")
    if baseline_hdd <= 0:
        raise ValueError("baseline_hdd must be positive")
    return round(eui_kwh_per_m2 * baseline_hdd / actual_hdd, 4)


def benchmark_eui(
    annual_kwh: float,
    floor_area_m2: float,
    sector: str,
) -> EUIResult:
    """Benchmark a building's EUI against its sector.

    Args:
        annual_kwh: Total annual energy consumption in kWh.
        floor_area_m2: Gross floor area in m².
        sector: Building sector key (see :data:`SECTOR_BENCHMARKS_KWH_PER_M2`).

    Returns:
        Populated :class:`EUIResult`.

    Raises:
        ValueError: If sector is unknown, annual_kwh < 0, or floor_area_m2 <= 0.
    """
    sector = sector.lower().replace(" ", "_")
    if sector not in SECTOR_BENCHMARKS_KWH_PER_M2:
        available = ", ".join(sorted(SECTOR_BENCHMARKS_KWH_PER_M2))
        raise ValueError(f"Unknown sector '{sector}'. Available: {available}")
    eui = energy_use_intensity(annual_kwh, floor_area_m2)
    benchmark = SECTOR_BENCHMARKS_KWH_PER_M2[sector]
    ratio = eui / benchmark
    pct_above = round((ratio - 1.0) * 100.0, 2)
    if ratio <= 0.5:
        rating = "A+"
    elif ratio <= 0.7:
        rating = "A"
    elif ratio <= 0.85:
        rating = "B"
    elif ratio <= 1.0:
        rating = "C"
    elif ratio <= 1.25:
        rating = "D"
    elif ratio <= 1.5:
        rating = "E"
    else:
        rating = "F"

    return EUIResult(
        annual_kwh=annual_kwh,
        floor_area_m2=floor_area_m2,
        eui_kwh_per_m2=eui,
        sector=sector,
        sector_benchmark_kwh_per_m2=benchmark,
        performance_ratio=round(ratio, 4),
        percent_above_benchmark=pct_above,
        rating=rating,
    )


def rank_within_sector(
    asset_id: str,
    eui_kwh_per_m2: float,
    peer_euis: list[float],
    sector: str,
) -> SectorRanking:
    """Rank a building's EUI within a peer group.

    Args:
        asset_id: Identifier for the building being ranked.
        eui_kwh_per_m2: Building's EUI in kWh/m²/year.
        peer_euis: List of peer EUIs in kWh/m²/year (including this building).
        sector: Building sector for context.

    Returns:
        Populated :class:`SectorRanking`.

    Raises:
        ValueError: If peer_euis is empty or eui_kwh_per_m2 <= 0.
    """
    if not peer_euis:
        raise ValueError("peer_euis must not be empty")
    if eui_kwh_per_m2 <= 0:
        raise ValueError("eui_kwh_per_m2 must be positive")
    sorted_euis = sorted(peer_euis)
    n = len(sorted_euis)
    better_count = sum(1 for p in peer_euis if p > eui_kwh_per_m2)
    rank = n - better_count
    percentile = round(100.0 * better_count / n, 2)
    return SectorRanking(
        asset_id=asset_id,
        eui_kwh_per_m2=eui_kwh_per_m2,
        sector=sector,
        percentile=percentile,
        rank=rank,
        total_peers=n,
        energy_star_eligible=percentile >= ENERGY_STAR_THRESHOLD_PERCENTILE,
    )


def savings_potential(
    current_eui: float,
    target_eui: float,
    floor_area_m2: float,
    energy_cost_per_kwh: float = 0.15,
) -> dict[str, float]:
    """Estimate annual energy and cost savings to reach a target EUI.

    Args:
        current_eui: Current EUI in kWh/m²/year.
        target_eui: Target EUI in kWh/m²/year.
        floor_area_m2: Floor area in m².
        energy_cost_per_kwh: Average energy cost in $/kWh.

    Returns:
        Dict with keys ``energy_kwh``, ``cost_dollars``, ``co2_kg``.

    Raises:
        ValueError: If target_eui >= current_eui or floor_area_m2 <= 0.
    """
    if target_eui >= current_eui:
        raise ValueError("target_eui must be less than current_eui")
    if floor_area_m2 <= 0:
        raise ValueError("floor_area_m2 must be positive")
    delta_eui = current_eui - target_eui
    energy_kwh = delta_eui * floor_area_m2
    return {
        "energy_kwh": round(energy_kwh, 2),
        "cost_dollars": round(energy_kwh * energy_cost_per_kwh, 2),
        "co2_kg": round(energy_kwh * 0.233, 2),
    }


def floor_area_from_eui(
    total_energy_kwh: float,
    eui_kwh_per_m2: float,
) -> float:
    """Return the floor area implied by a total energy consumption and EUI.

    Args:
        total_energy_kwh: Total annual energy consumption in kWh.
        eui_kwh_per_m2: Energy use intensity in kWh/m² (must be positive).

    Returns:
        Floor area in m².

    Raises:
        ValueError: If total_energy_kwh < 0 or eui_kwh_per_m2 ≤ 0.
    """
    if total_energy_kwh < 0:
        raise ValueError("total_energy_kwh must be non-negative")
    if eui_kwh_per_m2 <= 0:
        raise ValueError("eui_kwh_per_m2 must be positive")
    return total_energy_kwh / eui_kwh_per_m2


def carbon_from_eui(
    eui_kwh_per_m2: float,
    floor_area_m2: float,
    carbon_intensity_kg_per_kwh: float = 0.233,
) -> float:
    """Return annual carbon emissions in kg CO2e from EUI and floor area.

    Args:
        eui_kwh_per_m2: Energy use intensity in kWh/m².
        floor_area_m2: Floor area in m² (must be positive).
        carbon_intensity_kg_per_kwh: Grid carbon intensity in kg CO2e/kWh.

    Returns:
        Annual carbon emissions in kg CO2e.

    Raises:
        ValueError: If eui_kwh_per_m2 < 0 or floor_area_m2 ≤ 0.
    """
    if eui_kwh_per_m2 < 0:
        raise ValueError("eui_kwh_per_m2 must be non-negative")
    if floor_area_m2 <= 0:
        raise ValueError("floor_area_m2 must be positive")
    return eui_kwh_per_m2 * floor_area_m2 * carbon_intensity_kg_per_kwh


def eui_reduction_needed(
    current_eui: float,
    target_eui: float,
) -> float:
    """Return the percentage reduction required to reach a target EUI.

    Args:
        current_eui: Current EUI in kWh/m² (must be positive).
        target_eui: Target EUI in kWh/m² (must be non-negative and ≤ current).

    Returns:
        Required percentage reduction as a value in [0, 100].

    Raises:
        ValueError: If current_eui ≤ 0 or target_eui > current_eui.
    """
    if current_eui <= 0:
        raise ValueError("current_eui must be positive")
    if target_eui > current_eui:
        raise ValueError("target_eui cannot exceed current_eui")
    return (1.0 - target_eui / current_eui) * 100.0
