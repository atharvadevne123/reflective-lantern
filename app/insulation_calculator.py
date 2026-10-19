"""Building insulation performance and heat loss calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

HOURS_PER_YEAR: int = 8760


class ConstructionType(str, Enum):
    """Common building construction types with indicative U-values."""

    SOLID_WALL_UNINSULATED = "solid_wall_uninsulated"
    CAVITY_WALL_UNINSULATED = "cavity_wall_uninsulated"
    CAVITY_WALL_INSULATED = "cavity_wall_insulated"
    SOLID_WALL_INSULATED = "solid_wall_insulated"
    ROOF_UNINSULATED = "roof_uninsulated"
    ROOF_100MM = "roof_100mm"
    ROOF_270MM = "roof_270mm"
    FLOOR_UNINSULATED = "floor_uninsulated"
    FLOOR_INSULATED = "floor_insulated"


DEFAULT_U_VALUES: dict[ConstructionType, float] = {
    ConstructionType.SOLID_WALL_UNINSULATED: 2.1,
    ConstructionType.CAVITY_WALL_UNINSULATED: 1.5,
    ConstructionType.CAVITY_WALL_INSULATED: 0.27,
    ConstructionType.SOLID_WALL_INSULATED: 0.30,
    ConstructionType.ROOF_UNINSULATED: 2.3,
    ConstructionType.ROOF_100MM: 0.35,
    ConstructionType.ROOF_270MM: 0.16,
    ConstructionType.FLOOR_UNINSULATED: 0.70,
    ConstructionType.FLOOR_INSULATED: 0.25,
}


@dataclass
class InsulationUpgradeResult:
    """Result of an insulation upgrade analysis."""

    existing_heat_loss_w: float
    upgraded_heat_loss_w: float
    heat_loss_reduction_w: float
    annual_energy_saved_kwh: float
    annual_cost_saved: float
    simple_payback_years: float
    co2_saved_kg: float


def heat_loss_watts(
    u_value_w_per_m2k: float,
    area_m2: float,
    delta_t_k: float,
) -> float:
    """Compute steady-state conductive heat loss through a building element.

    Args:
        u_value_w_per_m2k: Thermal transmittance in W/m²·K.
        area_m2: Element area in m².
        delta_t_k: Indoor-outdoor temperature difference in K.

    Returns:
        Heat loss in Watts.

    Raises:
        ValueError: If any argument is non-positive.
    """
    if u_value_w_per_m2k <= 0:
        raise ValueError("u_value_w_per_m2k must be positive")
    if area_m2 <= 0:
        raise ValueError("area_m2 must be positive")
    if delta_t_k < 0:
        raise ValueError("delta_t_k must be non-negative")
    return round(u_value_w_per_m2k * area_m2 * delta_t_k, 4)


def r_value_from_u(u_value: float) -> float:
    """Convert U-value to R-value (thermal resistance).

    Args:
        u_value: Thermal transmittance in W/m²·K.

    Returns:
        Thermal resistance in m²·K/W.

    Raises:
        ValueError: If u_value is non-positive.
    """
    if u_value <= 0:
        raise ValueError("u_value must be positive")
    return round(1.0 / u_value, 6)


def u_value_from_r(r_value: float) -> float:
    """Convert R-value to U-value.

    Args:
        r_value: Thermal resistance in m²·K/W.

    Returns:
        Thermal transmittance in W/m²·K.

    Raises:
        ValueError: If r_value is non-positive.
    """
    if r_value <= 0:
        raise ValueError("r_value must be positive")
    return round(1.0 / r_value, 6)


def combined_u_value(layers: list[float]) -> float:
    """Compute the overall U-value for a composite wall with multiple layers.

    Args:
        layers: List of R-values for each layer in m²·K/W.

    Returns:
        Combined U-value in W/m²·K.

    Raises:
        ValueError: If layers is empty or contains non-positive values.
    """
    if not layers:
        raise ValueError("layers must not be empty")
    if any(r <= 0 for r in layers):
        raise ValueError("All layer R-values must be positive")
    total_r = sum(layers)
    return round(1.0 / total_r, 6)


def annual_heat_loss_kwh(
    heat_loss_w: float,
    heating_hours_per_year: float = 4380.0,
) -> float:
    """Estimate annual energy lost through a building element.

    Args:
        heat_loss_w: Peak heat loss in Watts.
        heating_hours_per_year: Annual heating hours (default 4380 = half year).

    Returns:
        Annual heat energy lost in kWh.

    Raises:
        ValueError: If arguments are non-positive.
    """
    if heat_loss_w < 0:
        raise ValueError("heat_loss_w must be non-negative")
    if heating_hours_per_year < 0:
        raise ValueError("heating_hours_per_year must be non-negative")
    return round(heat_loss_w * heating_hours_per_year / 1000.0, 4)


def analyse_insulation_upgrade(
    area_m2: float,
    existing_u: float,
    upgraded_u: float,
    delta_t_k: float,
    installation_cost: float,
    energy_rate: float = 0.10,
    heating_hours: float = 4380.0,
    emission_factor_kg_per_kwh: float = 0.233,
) -> InsulationUpgradeResult:
    """Analyse the cost-benefit of an insulation upgrade.

    Args:
        area_m2: Element area in m².
        existing_u: Current U-value in W/m²·K.
        upgraded_u: Post-upgrade U-value in W/m²·K.
        delta_t_k: Design temperature difference in K.
        installation_cost: Cost of the upgrade in currency units.
        energy_rate: Energy unit cost in currency/kWh.
        heating_hours: Annual heating hours.
        emission_factor_kg_per_kwh: Grid emission intensity.

    Returns:
        Populated :class:`InsulationUpgradeResult`.

    Raises:
        ValueError: If upgraded_u is not lower than existing_u.
    """
    if upgraded_u >= existing_u:
        raise ValueError("upgraded_u must be lower than existing_u for improvement")
    existing_w = heat_loss_watts(existing_u, area_m2, delta_t_k)
    upgraded_w = heat_loss_watts(upgraded_u, area_m2, delta_t_k)
    reduction_w = round(existing_w - upgraded_w, 4)
    existing_kwh = annual_heat_loss_kwh(existing_w, heating_hours)
    upgraded_kwh = annual_heat_loss_kwh(upgraded_w, heating_hours)
    saved_kwh = round(existing_kwh - upgraded_kwh, 4)
    cost_saved = round(saved_kwh * energy_rate, 2)
    payback = round(installation_cost / cost_saved, 2) if cost_saved > 0 else float("inf")
    co2_saved = round(saved_kwh * emission_factor_kg_per_kwh, 2)
    return InsulationUpgradeResult(
        existing_heat_loss_w=existing_w,
        upgraded_heat_loss_w=upgraded_w,
        heat_loss_reduction_w=reduction_w,
        annual_energy_saved_kwh=saved_kwh,
        annual_cost_saved=cost_saved,
        simple_payback_years=payback,
        co2_saved_kg=co2_saved,
    )
