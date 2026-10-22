"""EU-style energy efficiency label assignment for buildings and appliances."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class EnergyClass(StrEnum):
    """EU energy efficiency class labels (A+++ best, G worst)."""

    A_TRIPLE_PLUS = "A+++"
    A_DOUBLE_PLUS = "A++"
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


BUILDING_EPC_THRESHOLDS: list[tuple[float, EnergyClass]] = [
    (50.0, EnergyClass.A),
    (100.0, EnergyClass.B),
    (150.0, EnergyClass.C),
    (200.0, EnergyClass.D),
    (250.0, EnergyClass.E),
    (350.0, EnergyClass.F),
]


@dataclass
class EnergyLabel:
    """Assigned energy efficiency label with supporting metrics."""

    label: EnergyClass
    eui_kwh_per_m2: float
    reference_eui: float
    efficiency_ratio: float
    co2_kg_per_m2: float
    recommended_actions: list[str]


def classify_building(
    eui_kwh_per_m2: float,
    thresholds: list[tuple[float, EnergyClass]] | None = None,
) -> EnergyClass:
    """Assign an EPC energy label based on Energy Use Intensity.

    Args:
        eui_kwh_per_m2: Annual energy use intensity in kWh/m².
        thresholds: Ordered (upper_bound, label) pairs; defaults to
            :data:`BUILDING_EPC_THRESHOLDS`.

    Returns:
        :class:`EnergyClass` for the given EUI.

    Raises:
        ValueError: If eui_kwh_per_m2 is negative.
    """
    if eui_kwh_per_m2 < 0:
        raise ValueError("eui_kwh_per_m2 must be non-negative")
    if thresholds is None:
        thresholds = BUILDING_EPC_THRESHOLDS
    for upper, label in thresholds:
        if eui_kwh_per_m2 <= upper:
            return label
    return EnergyClass.G


def efficiency_ratio(eui: float, reference_eui: float) -> float:
    """Compute EUI relative to a reference benchmark.

    Args:
        eui: Actual EUI in kWh/m².
        reference_eui: Benchmark EUI for the building type in kWh/m².

    Returns:
        Ratio eui / reference_eui rounded to 4 decimal places.
        Returns 0.0 if reference_eui is zero.

    Raises:
        ValueError: If either argument is negative.
    """
    if eui < 0 or reference_eui < 0:
        raise ValueError("EUI values must be non-negative")
    if reference_eui == 0.0:
        return 0.0
    return round(eui / reference_eui, 4)


def label_building(
    floor_area_m2: float,
    annual_kwh: float,
    reference_eui: float = 200.0,
    grid_intensity_kg_per_kwh: float = 0.233,
) -> EnergyLabel:
    """Build a complete :class:`EnergyLabel` for a building.

    Args:
        floor_area_m2: Total conditioned floor area in m².
        annual_kwh: Annual energy consumption in kWh.
        reference_eui: Benchmark EUI for this building type (kWh/m²).
        grid_intensity_kg_per_kwh: Grid CO2 intensity for CO2 calculation.

    Returns:
        Populated :class:`EnergyLabel`.

    Raises:
        ValueError: If floor_area_m2 <= 0 or annual_kwh is negative.
    """
    if floor_area_m2 <= 0:
        raise ValueError("floor_area_m2 must be positive")
    if annual_kwh < 0:
        raise ValueError("annual_kwh must be non-negative")
    eui = annual_kwh / floor_area_m2
    label = classify_building(eui)
    ratio = efficiency_ratio(eui, reference_eui)
    co2 = round(eui * grid_intensity_kg_per_kwh, 4)
    actions = _recommend_actions(label, eui)
    return EnergyLabel(
        label=label,
        eui_kwh_per_m2=round(eui, 4),
        reference_eui=reference_eui,
        efficiency_ratio=ratio,
        co2_kg_per_m2=co2,
        recommended_actions=actions,
    )


def _recommend_actions(label: EnergyClass, eui: float) -> list[str]:
    """Return a prioritised list of improvement recommendations."""
    if label in (
        EnergyClass.A,
        EnergyClass.A_PLUS,
        EnergyClass.A_DOUBLE_PLUS,
        EnergyClass.A_TRIPLE_PLUS,
    ):
        return ["Maintain current efficiency; consider on-site renewable generation"]
    recs: list[str] = []
    if eui > 250:
        recs.append("Upgrade HVAC system to high-efficiency heat pump")
    if eui > 150:
        recs.append("Improve building envelope insulation")
        recs.append("Install LED lighting with occupancy sensors")
    if eui > 100:
        recs.append("Implement building energy management system (BEMS)")
    recs.append("Commission an in-depth energy audit")
    return recs


def compare_buildings(
    buildings: list[tuple[str, float, float]],
    reference_eui: float = 200.0,
) -> list[tuple[str, EnergyLabel]]:
    """Label multiple buildings and return them sorted best-to-worst.

    Args:
        buildings: List of (name, floor_area_m2, annual_kwh) tuples.
        reference_eui: Common benchmark EUI.

    Returns:
        List of (name, :class:`EnergyLabel`) sorted by ascending EUI.
    """
    labelled = [(name, label_building(area, kwh, reference_eui)) for name, area, kwh in buildings]
    return sorted(labelled, key=lambda x: x[1].eui_kwh_per_m2)
