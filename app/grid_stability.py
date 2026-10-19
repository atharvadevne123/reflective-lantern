"""Grid stability metrics and power factor analysis for energy systems."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_NOMINAL_VOLTAGE: float = 230.0
"""Nominal voltage in Volts for European residential grid."""

DEFAULT_FREQUENCY_HZ: float = 50.0
"""Nominal AC frequency in Hz."""

VOLTAGE_TOLERANCE: float = 0.10
"""EN 50160 standard: ±10% of nominal voltage is acceptable."""

FREQUENCY_TOLERANCE_HZ: float = 0.5
"""Normal frequency deviation band in Hz."""


@dataclass
class GridMetrics:
    """Aggregated grid stability metrics for a measurement period."""

    voltage_v: float
    frequency_hz: float
    active_power_kw: float
    reactive_power_kvar: float
    apparent_power_kva: float
    power_factor: float
    thd_percent: float
    voltage_unbalance_percent: float
    is_voltage_normal: bool
    is_frequency_normal: bool


def power_factor(
    active_kw: float,
    reactive_kvar: float,
) -> float:
    """Compute displacement power factor from active and reactive power.

    Args:
        active_kw: Active (real) power in kW.
        reactive_kvar: Reactive power in kVAR (positive = inductive).

    Returns:
        Power factor in [0, 1].  Returns 0.0 when apparent power is zero.

    Raises:
        ValueError: If either argument is negative.
    """
    if active_kw < 0 or reactive_kvar < 0:
        raise ValueError("active_kw and reactive_kvar must be non-negative")
    apparent = math.hypot(active_kw, reactive_kvar)
    if apparent == 0.0:
        return 0.0
    return round(active_kw / apparent, 6)


def apparent_power_kva(active_kw: float, reactive_kvar: float) -> float:
    """Return apparent power in kVA from active and reactive components.

    Args:
        active_kw: Active power in kW.
        reactive_kvar: Reactive power in kVAR.

    Returns:
        Apparent power in kVA rounded to 4 decimal places.
    """
    return round(math.hypot(active_kw, reactive_kvar), 4)


def reactive_power_from_pf(active_kw: float, pf: float) -> float:
    """Derive reactive power from active power and target power factor.

    Args:
        active_kw: Active power in kW (non-negative).
        pf: Power factor in (0, 1].

    Returns:
        Reactive power in kVAR rounded to 4 decimal places.

    Raises:
        ValueError: If pf is outside (0, 1] or active_kw is negative.
    """
    if not (0 < pf <= 1):
        raise ValueError("pf must be in (0, 1]")
    if active_kw < 0:
        raise ValueError("active_kw must be non-negative")
    angle = math.acos(pf)
    return round(active_kw * math.tan(angle), 4)


def voltage_unbalance_percent(
    phase_a_v: float,
    phase_b_v: float,
    phase_c_v: float,
) -> float:
    """Compute NEMA voltage unbalance percentage.

    Defined as 100 * max_deviation / average.

    Args:
        phase_a_v: Phase A voltage in Volts.
        phase_b_v: Phase B voltage in Volts.
        phase_c_v: Phase C voltage in Volts.

    Returns:
        Unbalance percentage rounded to 3 decimal places.

    Raises:
        ValueError: If any voltage is negative.
    """
    for v in (phase_a_v, phase_b_v, phase_c_v):
        if v < 0:
            raise ValueError("Phase voltages must be non-negative")
    avg = (phase_a_v + phase_b_v + phase_c_v) / 3.0
    if avg == 0.0:
        return 0.0
    max_dev = max(abs(v - avg) for v in (phase_a_v, phase_b_v, phase_c_v))
    return round(100.0 * max_dev / avg, 3)


def is_voltage_normal(
    voltage_v: float,
    nominal_v: float = DEFAULT_NOMINAL_VOLTAGE,
    tolerance: float = VOLTAGE_TOLERANCE,
) -> bool:
    """Check whether a voltage reading is within the acceptable band.

    Args:
        voltage_v: Measured voltage in Volts.
        nominal_v: Nominal voltage in Volts.
        tolerance: Fractional tolerance (default ±10%).

    Returns:
        True if within [nominal*(1-tol), nominal*(1+tol)].
    """
    low = nominal_v * (1.0 - tolerance)
    high = nominal_v * (1.0 + tolerance)
    return low <= voltage_v <= high


def is_frequency_normal(
    frequency_hz: float,
    nominal_hz: float = DEFAULT_FREQUENCY_HZ,
    tolerance_hz: float = FREQUENCY_TOLERANCE_HZ,
) -> bool:
    """Check whether a frequency reading is within the normal band.

    Args:
        frequency_hz: Measured frequency in Hz.
        nominal_hz: Nominal frequency in Hz.
        tolerance_hz: Absolute tolerance in Hz (default ±0.5 Hz).

    Returns:
        True if within [nominal-tol, nominal+tol].
    """
    return abs(frequency_hz - nominal_hz) <= tolerance_hz


def thd_percent(harmonics: list[float], fundamental: float) -> float:
    """Compute Total Harmonic Distortion as a percentage.

    Args:
        harmonics: RMS magnitudes of 2nd, 3rd, ... harmonic components.
        fundamental: RMS magnitude of the fundamental component.

    Returns:
        THD as a percentage rounded to 3 decimal places.

    Raises:
        ValueError: If fundamental is zero or any value is negative.
    """
    if fundamental <= 0:
        raise ValueError("fundamental must be positive")
    for h in harmonics:
        if h < 0:
            raise ValueError("Harmonic magnitudes must be non-negative")
    rms_harmonics = math.sqrt(sum(h**2 for h in harmonics))
    return round(100.0 * rms_harmonics / fundamental, 3)


def grid_metrics(
    voltage_v: float,
    frequency_hz: float,
    active_kw: float,
    reactive_kvar: float,
    harmonics: list[float] | None = None,
    phase_voltages: tuple[float, float, float] | None = None,
) -> GridMetrics:
    """Build a :class:`GridMetrics` snapshot from raw measurements.

    Args:
        voltage_v: Line voltage in Volts.
        frequency_hz: Supply frequency in Hz.
        active_kw: Active power in kW.
        reactive_kvar: Reactive power in kVAR.
        harmonics: Optional list of harmonic RMS magnitudes (2nd, 3rd, ...).
        phase_voltages: Optional (Va, Vb, Vc) tuple for unbalance calculation.

    Returns:
        A populated :class:`GridMetrics` dataclass.
    """
    pf = power_factor(abs(active_kw), abs(reactive_kvar))
    apparent = apparent_power_kva(abs(active_kw), abs(reactive_kvar))
    thd = thd_percent(harmonics, voltage_v) if harmonics else 0.0
    unbalance = (
        voltage_unbalance_percent(*phase_voltages)
        if phase_voltages is not None
        else 0.0
    )
    return GridMetrics(
        voltage_v=voltage_v,
        frequency_hz=frequency_hz,
        active_power_kw=active_kw,
        reactive_power_kvar=reactive_kvar,
        apparent_power_kva=apparent,
        power_factor=pf,
        thd_percent=thd,
        voltage_unbalance_percent=unbalance,
        is_voltage_normal=is_voltage_normal(voltage_v),
        is_frequency_normal=is_frequency_normal(frequency_hz),
    )
