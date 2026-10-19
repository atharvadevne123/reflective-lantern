"""Electrical cable sizing and voltage drop calculation utilities."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

COPPER_RESISTIVITY: float = 1.72e-8
"""Resistivity of copper at 20°C in Ω·m."""

ALUMINIUM_RESISTIVITY: float = 2.82e-8
"""Resistivity of aluminium at 20°C in Ω·m."""

MAX_VOLTAGE_DROP_PCT: float = 3.0
"""IEC 60364 maximum voltage drop for final circuits (%)."""


@dataclass
class CableResult:
    """Result of a cable voltage drop calculation."""

    resistance_ohm: float
    reactance_ohm: float
    impedance_ohm: float
    voltage_drop_v: float
    voltage_drop_pct: float
    within_limit: bool
    power_loss_w: float


def cable_resistance(
    length_m: float,
    cross_section_mm2: float,
    resistivity: float = COPPER_RESISTIVITY,
    temperature_c: float = 20.0,
) -> float:
    """Compute DC resistance of a cable run.

    Args:
        length_m: One-way cable length in metres.
        cross_section_mm2: Conductor cross-sectional area in mm².
        resistivity: Material resistivity in Ω·m (default copper).
        temperature_c: Operating temperature in °C.

    Returns:
        Round-trip cable resistance in Ω.

    Raises:
        ValueError: If any argument is non-positive.
    """
    if length_m <= 0:
        raise ValueError("length_m must be positive")
    if cross_section_mm2 <= 0:
        raise ValueError("cross_section_mm2 must be positive")
    if resistivity <= 0:
        raise ValueError("resistivity must be positive")
    area_m2 = cross_section_mm2 * 1e-6
    temp_factor = 1 + 0.00393 * (temperature_c - 20.0)
    return round(2 * resistivity * length_m * temp_factor / area_m2, 6)


def voltage_drop_single_phase(
    current_a: float,
    resistance_ohm: float,
    power_factor: float = 1.0,
    reactance_ohm: float = 0.0,
) -> float:
    """Compute voltage drop for a single-phase circuit.

    Args:
        current_a: Load current in A.
        resistance_ohm: Cable resistance in Ω.
        power_factor: Load power factor (0–1).
        reactance_ohm: Cable reactance in Ω.

    Returns:
        Voltage drop in V.

    Raises:
        ValueError: If current is negative or power factor out of (0, 1].
    """
    if current_a < 0:
        raise ValueError("current_a must be non-negative")
    if not (0 < power_factor <= 1):
        raise ValueError("power_factor must be in (0, 1]")
    pf_angle = math.acos(power_factor)
    r_component = current_a * resistance_ohm * power_factor
    x_component = current_a * reactance_ohm * math.sin(pf_angle)
    return round(r_component + x_component, 4)


def voltage_drop_three_phase(
    current_a: float,
    resistance_ohm: float,
    power_factor: float = 1.0,
    reactance_ohm: float = 0.0,
) -> float:
    """Compute voltage drop for a balanced three-phase circuit.

    Args:
        current_a: Phase current in A.
        resistance_ohm: Per-phase cable resistance in Ω.
        power_factor: Load power factor.
        reactance_ohm: Per-phase cable reactance in Ω.

    Returns:
        Line-to-line voltage drop in V.

    Raises:
        ValueError: If arguments are invalid.
    """
    if current_a < 0:
        raise ValueError("current_a must be non-negative")
    if not (0 < power_factor <= 1):
        raise ValueError("power_factor must be in (0, 1]")
    pf_angle = math.acos(power_factor)
    r_component = current_a * resistance_ohm * power_factor
    x_component = current_a * reactance_ohm * math.sin(pf_angle)
    return round(math.sqrt(3) * (r_component + x_component), 4)


def voltage_drop_percent(
    voltage_drop_v: float,
    nominal_voltage_v: float,
) -> float:
    """Compute voltage drop as a percentage of nominal.

    Args:
        voltage_drop_v: Absolute voltage drop in V.
        nominal_voltage_v: Nominal supply voltage in V.

    Returns:
        Voltage drop in %.

    Raises:
        ValueError: If nominal_voltage_v is non-positive.
    """
    if nominal_voltage_v <= 0:
        raise ValueError("nominal_voltage_v must be positive")
    if voltage_drop_v < 0:
        raise ValueError("voltage_drop_v must be non-negative")
    return round(voltage_drop_v / nominal_voltage_v * 100.0, 4)


def analyse_cable(
    length_m: float,
    current_a: float,
    cross_section_mm2: float,
    nominal_voltage_v: float,
    power_factor: float = 1.0,
    is_three_phase: bool = False,
    reactance_per_km: float = 0.08,
) -> CableResult:
    """Run a complete cable voltage drop analysis.

    Args:
        length_m: Cable run length in metres.
        current_a: Load current in A.
        cross_section_mm2: Conductor cross-section in mm².
        nominal_voltage_v: Nominal supply voltage in V.
        power_factor: Load power factor.
        is_three_phase: True for three-phase, False for single-phase.
        reactance_per_km: Cable reactance in Ω/km (default 0.08 for copper).

    Returns:
        Populated :class:`CableResult`.
    """
    resistance = cable_resistance(length_m, cross_section_mm2)
    reactance = reactance_per_km * 2 * length_m / 1000.0
    impedance = round(math.sqrt(resistance**2 + reactance**2), 6)
    if is_three_phase:
        vd = voltage_drop_three_phase(current_a, resistance, power_factor, reactance)
    else:
        vd = voltage_drop_single_phase(current_a, resistance, power_factor, reactance)
    vd_pct = voltage_drop_percent(vd, nominal_voltage_v)
    power_loss = round(current_a**2 * resistance, 4)
    return CableResult(
        resistance_ohm=resistance,
        reactance_ohm=reactance,
        impedance_ohm=impedance,
        voltage_drop_v=vd,
        voltage_drop_pct=vd_pct,
        within_limit=vd_pct <= MAX_VOLTAGE_DROP_PCT,
        power_loss_w=power_loss,
    )
