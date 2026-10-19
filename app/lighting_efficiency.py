"""Lighting energy efficiency calculations: lumen efficacy, daylight integration."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

INCANDESCENT_EFFICACY_LM_PER_W: float = 15.0
"""Typical incandescent bulb efficacy (lm/W)."""

HALOGEN_EFFICACY_LM_PER_W: float = 20.0
"""Typical halogen lamp efficacy (lm/W)."""

CFL_EFFICACY_LM_PER_W: float = 60.0
"""Typical compact fluorescent lamp efficacy (lm/W)."""

LED_EFFICACY_LM_PER_W: float = 130.0
"""Modern LED lamp efficacy (lm/W)."""

ILLUMINANCE_OFFICE_LUX: float = 500.0
"""CIBSE recommended maintained illuminance for office tasks (lux)."""


@dataclass
class LightingAuditResult:
    """Lighting system energy and cost audit result."""

    installed_watts: float
    annual_kwh: float
    annual_cost: float
    lux_per_watt: float
    efficacy_lm_per_w: float
    upgrade_savings_kwh: float | None
    upgrade_payback_years: float | None


def luminous_efficacy(
    lumens: float,
    watts: float,
) -> float:
    """Compute luminous efficacy in lm/W.

    Args:
        lumens: Luminous flux output in lumens.
        watts: Power consumption in watts.

    Returns:
        Efficacy in lm/W.

    Raises:
        ValueError: If watts <= 0 or lumens < 0.
    """
    if watts <= 0:
        raise ValueError("watts must be positive")
    if lumens < 0:
        raise ValueError("lumens must be non-negative")
    return round(lumens / watts, 4)


def power_for_target_lux(
    area_m2: float,
    target_lux: float,
    maintenance_factor: float,
    utilisation_factor: float,
    efficacy_lm_per_w: float = LED_EFFICACY_LM_PER_W,
) -> float:
    """Estimate installed wattage required to achieve a target illuminance.

    Uses the lumen method: N × Φ × MF × UF = E × A
    where Φ is the per-lamp output derived from wattage and efficacy.

    Args:
        area_m2: Room floor area in m².
        target_lux: Required maintained illuminance in lux.
        maintenance_factor: Luminaire dirt/ageing factor (0–1).
        utilisation_factor: Room geometry utilisation factor (0–1).
        efficacy_lm_per_w: Lamp efficacy in lm/W.

    Returns:
        Installed power in watts.

    Raises:
        ValueError: If any factor is outside (0, 1] or other args are non-positive.
    """
    if area_m2 <= 0 or target_lux <= 0:
        raise ValueError("area_m2 and target_lux must be positive")
    if not (0 < maintenance_factor <= 1) or not (0 < utilisation_factor <= 1):
        raise ValueError("maintenance_factor and utilisation_factor must be in (0, 1]")
    if efficacy_lm_per_w <= 0:
        raise ValueError("efficacy_lm_per_w must be positive")
    required_lumens = (target_lux * area_m2) / (maintenance_factor * utilisation_factor)
    return round(required_lumens / efficacy_lm_per_w, 4)


def annual_kwh(
    installed_watts: float,
    daily_hours: float,
    occupancy_fraction: float = 1.0,
    days_per_year: int = 365,
) -> float:
    """Estimate annual lighting energy consumption.

    Args:
        installed_watts: Total installed lighting power in watts.
        daily_hours: Average operating hours per day.
        occupancy_fraction: Fraction of time the space is actually occupied (0–1).
        days_per_year: Calendar days per year.

    Returns:
        Annual energy in kWh.

    Raises:
        ValueError: If any argument is out of range.
    """
    if installed_watts < 0 or daily_hours < 0:
        raise ValueError("installed_watts and daily_hours must be non-negative")
    if not (0 <= occupancy_fraction <= 1):
        raise ValueError("occupancy_fraction must be in [0, 1]")
    return round(installed_watts * daily_hours * occupancy_fraction * days_per_year / 1000.0, 4)


def upgrade_savings(
    old_watts: float,
    new_watts: float,
    daily_hours: float,
    energy_rate: float,
    capex: float,
    days_per_year: int = 365,
) -> tuple[float, float]:
    """Compute energy savings and payback period for a lighting upgrade.

    Args:
        old_watts: Current installed wattage.
        new_watts: Proposed installed wattage.
        daily_hours: Daily operating hours.
        energy_rate: Energy cost in $/kWh.
        capex: Cost of the lighting upgrade.
        days_per_year: Operating days per year.

    Returns:
        Tuple of (annual_savings_kwh, payback_years).

    Raises:
        ValueError: If new_watts >= old_watts (no savings) or any arg is negative.
    """
    if old_watts < 0 or new_watts < 0 or daily_hours < 0 or energy_rate < 0 or capex < 0:
        raise ValueError("All arguments must be non-negative")
    if new_watts >= old_watts:
        raise ValueError("new_watts must be less than old_watts for a saving")
    saved_kwh = annual_kwh(old_watts - new_watts, daily_hours, days_per_year=days_per_year)
    saved_cost = saved_kwh * energy_rate
    payback = capex / saved_cost if saved_cost > 0 else float("inf")
    return round(saved_kwh, 4), round(payback, 4)


def audit_space(
    area_m2: float,
    installed_watts: float,
    daily_hours: float,
    energy_rate: float = 0.28,
    target_lux: float = ILLUMINANCE_OFFICE_LUX,
    current_lux: float | None = None,
    led_upgrade_cost_per_w: float = 3.0,
) -> LightingAuditResult:
    """Audit a space's lighting system and size an LED upgrade.

    Args:
        area_m2: Floor area in m².
        installed_watts: Current installed lighting power in watts.
        daily_hours: Operating hours per day.
        energy_rate: Energy cost in $/kWh.
        target_lux: Target maintained illuminance.
        current_lux: Current measured illuminance (optional).
        led_upgrade_cost_per_w: Installed cost of LED replacement in $/W.

    Returns:
        Populated :class:`LightingAuditResult`.
    """
    kwh = annual_kwh(installed_watts, daily_hours)
    cost = round(kwh * energy_rate, 4)
    lux_per_w = (current_lux / installed_watts) if current_lux and installed_watts > 0 else 0.0
    efficacy = round(installed_watts / area_m2, 4) if area_m2 > 0 else 0.0
    led_watts = power_for_target_lux(area_m2, target_lux, 0.80, 0.70)
    capex = led_watts * led_upgrade_cost_per_w
    try:
        saved_kwh, payback = upgrade_savings(installed_watts, led_watts, daily_hours, energy_rate, capex)
    except ValueError:
        saved_kwh, payback = None, None
    return LightingAuditResult(
        installed_watts=installed_watts,
        annual_kwh=kwh,
        annual_cost=cost,
        lux_per_watt=round(lux_per_w, 4),
        efficacy_lm_per_w=efficacy,
        upgrade_savings_kwh=saved_kwh,
        upgrade_payback_years=payback,
    )
