"""Electrical power-quality metrics.

Computes the quantities a facilities team watches on the supply side:
power factor, apparent and reactive power, voltage imbalance across phases,
and the capacitor sizing needed to correct a lagging power factor.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass

logger = logging.getLogger(__name__)

GOOD_POWER_FACTOR: float = 0.95
"""Power factor at or above which no correction is usually required."""

POOR_POWER_FACTOR: float = 0.85
"""Power factor below which utilities commonly levy a penalty."""

MAX_VOLTAGE_IMBALANCE_PCT: float = 2.0
"""NEMA guidance: motors should not run above 2% voltage imbalance."""


@dataclass
class PowerQualityReport:
    """Summary of supply-side power quality for one site."""

    real_power_kw: float
    apparent_power_kva: float
    reactive_power_kvar: float
    power_factor: float
    power_factor_rating: str
    voltage_imbalance_pct: float
    imbalance_within_limit: bool


def power_factor(real_power_kw: float, apparent_power_kva: float) -> float:
    """Return the ratio of real power to apparent power.

    Args:
        real_power_kw: Real (working) power in kW.
        apparent_power_kva: Apparent (total) power in kVA.

    Returns:
        Power factor in 0-1 rounded to 4 decimal places. Returns ``0.0``
        when *apparent_power_kva* is zero.

    Raises:
        ValueError: If either argument is negative, or real power exceeds
            apparent power (physically impossible).
    """
    if real_power_kw < 0 or apparent_power_kva < 0:
        raise ValueError(f"power must be non-negative, got real={real_power_kw} apparent={apparent_power_kva}")
    if apparent_power_kva == 0:
        return 0.0
    if real_power_kw > apparent_power_kva:
        raise ValueError(f"real power ({real_power_kw} kW) cannot exceed apparent power ({apparent_power_kva} kVA)")
    return round(real_power_kw / apparent_power_kva, 4)


def apparent_power(real_power_kw: float, reactive_power_kvar: float) -> float:
    """Return apparent power from the real and reactive components.

    Args:
        real_power_kw: Real (working) power in kW.
        reactive_power_kvar: Reactive power in kVAR.

    Returns:
        Apparent power in kVA rounded to 4 decimal places.

    Raises:
        ValueError: If *real_power_kw* is negative.
    """
    if real_power_kw < 0:
        raise ValueError(f"real_power_kw must be non-negative, got {real_power_kw}")
    return round(math.hypot(real_power_kw, reactive_power_kvar), 4)


def reactive_power(real_power_kw: float, power_factor_value: float) -> float:
    """Return reactive power implied by a real power and power factor.

    Args:
        real_power_kw: Real (working) power in kW.
        power_factor_value: Power factor in 0-1.

    Returns:
        Reactive power in kVAR rounded to 4 decimal places. Returns ``0.0``
        at unity power factor.

    Raises:
        ValueError: If *real_power_kw* is negative or *power_factor_value*
            is outside the open-closed interval (0, 1].
    """
    if real_power_kw < 0:
        raise ValueError(f"real_power_kw must be non-negative, got {real_power_kw}")
    if not 0.0 < power_factor_value <= 1.0:
        raise ValueError(f"power_factor_value must be in (0, 1], got {power_factor_value}")
    angle = math.acos(power_factor_value)
    return round(real_power_kw * math.tan(angle), 4)


def rate_power_factor(power_factor_value: float) -> str:
    """Bucket a power factor into a qualitative rating.

    Args:
        power_factor_value: Power factor in 0-1.

    Returns:
        One of ``"good"``, ``"acceptable"``, or ``"poor"``.
    """
    if power_factor_value >= GOOD_POWER_FACTOR:
        return "good"
    if power_factor_value >= POOR_POWER_FACTOR:
        return "acceptable"
    return "poor"


def correction_kvar(
    real_power_kw: float,
    current_power_factor: float,
    target_power_factor: float = GOOD_POWER_FACTOR,
) -> float:
    """Return the capacitor rating needed to reach a target power factor.

    Args:
        real_power_kw: Real (working) power in kW.
        current_power_factor: Present power factor in (0, 1].
        target_power_factor: Desired power factor in (0, 1].

    Returns:
        Required capacitor rating in kVAR rounded to 4 decimal places.
        Returns ``0.0`` when the current factor already meets the target.

    Raises:
        ValueError: If *real_power_kw* is negative or either power factor
            is outside (0, 1].
    """
    if not 0.0 < target_power_factor <= 1.0:
        raise ValueError(f"target_power_factor must be in (0, 1], got {target_power_factor}")
    if current_power_factor >= target_power_factor:
        return 0.0
    current_kvar = reactive_power(real_power_kw, current_power_factor)
    target_kvar = reactive_power(real_power_kw, target_power_factor)
    return round(current_kvar - target_kvar, 4)


def voltage_imbalance(phase_voltages: list[float]) -> float:
    """Return percentage voltage imbalance across supply phases.

    Uses the NEMA definition: the largest deviation of any phase from the
    mean, expressed as a percentage of the mean.

    Args:
        phase_voltages: Measured voltage on each phase.

    Returns:
        Imbalance percentage rounded to 4 decimal places. Returns ``0.0``
        when the mean voltage is zero.

    Raises:
        ValueError: If fewer than two phases are supplied or any voltage
            is negative.
    """
    if len(phase_voltages) < 2:
        raise ValueError(f"at least 2 phase voltages required, got {len(phase_voltages)}")
    for voltage in phase_voltages:
        if voltage < 0:
            raise ValueError(f"phase voltages must be non-negative, got {voltage}")

    mean_voltage = statistics.fmean(phase_voltages)
    if mean_voltage == 0:
        return 0.0
    max_deviation = max(abs(voltage - mean_voltage) for voltage in phase_voltages)
    return round(100.0 * max_deviation / mean_voltage, 4)


def build_report(
    real_power_kw: float,
    reactive_power_kvar: float,
    phase_voltages: list[float],
) -> PowerQualityReport:
    """Assemble a full power-quality report for one site.

    Args:
        real_power_kw: Real (working) power in kW.
        reactive_power_kvar: Reactive power in kVAR.
        phase_voltages: Measured voltage on each supply phase.

    Returns:
        A populated :class:`PowerQualityReport`.

    Raises:
        ValueError: If *real_power_kw* is negative, or fewer than two phase
            voltages are supplied.
    """
    kva = apparent_power(real_power_kw, reactive_power_kvar)
    pf = power_factor(real_power_kw, kva)
    imbalance = voltage_imbalance(phase_voltages)

    report = PowerQualityReport(
        real_power_kw=round(real_power_kw, 4),
        apparent_power_kva=kva,
        reactive_power_kvar=round(reactive_power_kvar, 4),
        power_factor=pf,
        power_factor_rating=rate_power_factor(pf),
        voltage_imbalance_pct=imbalance,
        imbalance_within_limit=imbalance <= MAX_VOLTAGE_IMBALANCE_PCT,
    )
    if report.power_factor_rating == "poor":
        logger.warning("Poor power factor %.3f — utility penalties likely", pf)
    if not report.imbalance_within_limit:
        logger.warning("Voltage imbalance %.2f%% exceeds the %.1f%% limit", imbalance, MAX_VOLTAGE_IMBALANCE_PCT)
    return report


__all__ = [
    "GOOD_POWER_FACTOR",
    "MAX_VOLTAGE_IMBALANCE_PCT",
    "POOR_POWER_FACTOR",
    "PowerQualityReport",
    "apparent_power",
    "build_report",
    "correction_kvar",
    "power_factor",
    "rate_power_factor",
    "reactive_power",
    "voltage_imbalance",
]
