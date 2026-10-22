"""Thermal comfort indices: PMV, PPD, and adaptive comfort models."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class ComfortCategory(StrEnum):
    """ASHRAE 55 / ISO 7730 thermal comfort acceptability categories."""

    CATEGORY_I = "I"
    CATEGORY_II = "II"
    CATEGORY_III = "III"
    OUT_OF_RANGE = "Out of range"


@dataclass
class ThermalComfortResult:
    """Thermal comfort assessment result."""

    pmv: float
    ppd: float
    category: ComfortCategory
    sensation: str


SENSATION_LABELS: dict[int, str] = {
    -3: "Cold",
    -2: "Cool",
    -1: "Slightly cool",
    0: "Neutral",
    1: "Slightly warm",
    2: "Warm",
    3: "Hot",
}


def pmv(
    air_temp_c: float,
    mean_radiant_temp_c: float,
    air_velocity_m_s: float,
    relative_humidity_pct: float,
    metabolic_rate_met: float = 1.2,
    clothing_insulation_clo: float = 1.0,
) -> float:
    """Compute ISO 7730 Predicted Mean Vote (PMV).

    Uses the Fanger model with a simplified heat-balance approach suitable
    for typical comfort calculations in steady-state conditions.

    Args:
        air_temp_c: Air (dry-bulb) temperature in °C.
        mean_radiant_temp_c: Mean radiant temperature in °C.
        air_velocity_m_s: Relative air speed in m/s (>= 0).
        relative_humidity_pct: Relative humidity in % (0–100).
        metabolic_rate_met: Metabolic rate in MET (1 MET = 58.15 W/m²).
        clothing_insulation_clo: Clothing insulation in CLO (1 CLO = 0.155 m²·K/W).

    Returns:
        PMV index in [-3, +3] (values outside indicate extreme conditions).

    Raises:
        ValueError: If air_velocity_m_s or relative_humidity_pct is out of range.
    """
    if air_velocity_m_s < 0:
        raise ValueError("air_velocity_m_s must be non-negative")
    if not (0 <= relative_humidity_pct <= 100):
        raise ValueError("relative_humidity_pct must be in [0, 100]")

    # Constants
    met = metabolic_rate_met
    clo = clothing_insulation_clo
    m = met * 58.15  # metabolic rate in W/m²
    w = 0.0  # external work (approx 0 for sedentary)
    icl = clo * 0.155  # clothing thermal resistance m²K/W

    # Clothing surface area factor
    fcl = 1 + 1.29 * icl if icl <= 0.078 else 1.05 + 0.645 * icl

    # Convective heat transfer coefficient
    hcf = 12.1 * math.sqrt(max(0.0, air_velocity_m_s))

    # Operative temperature approximation
    ta = air_temp_c
    tr = mean_radiant_temp_c

    # Water vapour partial pressure (Pa) via Magnus formula approximation
    pa = (relative_humidity_pct / 100.0) * math.exp(16.6536 - 4030.183 / (ta + 235.0))

    # Clothing surface temperature (iterative; approximate one-shot)
    tcl = ta + (35.5 - ta) / (3.5 * (6.45 * icl + 0.1))

    # Radiation + convection from clothing
    hc = max(hcf, 2.38 * abs(tcl - ta) ** 0.25)

    # Heat loss terms
    hl1 = 3.05e-3 * (5733 - 6.99 * (m - w) - pa)  # skin diffusion
    hl2 = max(0.0, 0.42 * ((m - w) - 58.15))  # sweating
    hl3 = 1.7e-5 * m * (5867 - pa)  # latent respiration
    hl4 = 0.0014 * m * (34.0 - ta)  # dry respiration
    hl5 = 3.96e-8 * fcl * ((tcl + 273) ** 4 - (tr + 273) ** 4)  # radiation
    hl6 = fcl * hc * (tcl - ta)  # convection

    ts = 0.303 * math.exp(-0.036 * m) + 0.028
    pmv_val = ts * ((m - w) - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    return round(pmv_val, 4)


def ppd(pmv_value: float) -> float:
    """Compute Predicted Percentage Dissatisfied (PPD) from PMV.

    Args:
        pmv_value: PMV index.

    Returns:
        PPD in percent in [5, 100].
    """
    val = 100.0 - 95.0 * math.exp(-0.03353 * pmv_value**4 - 0.2179 * pmv_value**2)
    return round(min(100.0, max(5.0, val)), 2)


def comfort_category(ppd_value: float) -> ComfortCategory:
    """Assign an ISO 7730 comfort category based on PPD.

    Args:
        ppd_value: Predicted percentage dissatisfied (%).

    Returns:
        :class:`ComfortCategory` (I = best, III = acceptable).
    """
    if ppd_value < 6:
        return ComfortCategory.CATEGORY_I
    if ppd_value < 10:
        return ComfortCategory.CATEGORY_II
    if ppd_value < 15:
        return ComfortCategory.CATEGORY_III
    return ComfortCategory.OUT_OF_RANGE


def sensation_label(pmv_value: float) -> str:
    """Convert a PMV value to a human-readable thermal sensation label.

    Args:
        pmv_value: PMV index.

    Returns:
        Sensation string from ASHRAE 55 7-point scale.
    """
    rounded = max(-3, min(3, round(pmv_value)))
    return SENSATION_LABELS.get(rounded, "Extreme")


def assess_comfort(
    air_temp_c: float,
    mean_radiant_temp_c: float,
    air_velocity_m_s: float,
    relative_humidity_pct: float,
    metabolic_rate_met: float = 1.2,
    clothing_insulation_clo: float = 1.0,
) -> ThermalComfortResult:
    """Compute a full :class:`ThermalComfortResult` for the given conditions.

    Wraps :func:`pmv`, :func:`ppd`, :func:`comfort_category`, and
    :func:`sensation_label` into a single convenience call.
    """
    pmv_val = pmv(
        air_temp_c,
        mean_radiant_temp_c,
        air_velocity_m_s,
        relative_humidity_pct,
        metabolic_rate_met,
        clothing_insulation_clo,
    )
    ppd_val = ppd(pmv_val)
    return ThermalComfortResult(
        pmv=pmv_val,
        ppd=ppd_val,
        category=comfort_category(ppd_val),
        sensation=sensation_label(pmv_val),
    )
