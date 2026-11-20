"""Power quality analysis: harmonics, sag/swell detection, and PQ indices."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)

NOMINAL_FREQUENCY_HZ: float = 50.0
NOMINAL_VOLTAGE_V: float = 230.0


class PQEvent(StrEnum):
    """Types of power quality disturbance events."""

    SAG = "sag"
    SWELL = "swell"
    INTERRUPTION = "interruption"
    NORMAL = "normal"


@dataclass
class PQEventRecord:
    """A detected power quality event."""

    event_type: PQEvent
    magnitude_pct: float
    duration_cycles: float
    timestamp_s: float = 0.0


@dataclass
class PQReport:
    """Summary power quality report for a monitoring period."""

    vrms_v: float
    thd_pct: float
    crest_factor: float
    k_factor: float
    events: list[PQEventRecord]
    within_en50160: bool


def rms_voltage(samples: list[float]) -> float:
    """Compute RMS voltage from instantaneous samples.

    Args:
        samples: List of instantaneous voltage samples in V.

    Returns:
        RMS voltage in V.

    Raises:
        ValueError: If samples is empty.
    """
    if not samples:
        raise ValueError("samples must not be empty")
    return round(math.sqrt(sum(s**2 for s in samples) / len(samples)), 4)


def thd_voltage(
    fundamental_v: float,
    harmonics_v: list[float],
) -> float:
    """Compute Total Harmonic Distortion (THD) as a percentage.

    Args:
        fundamental_v: RMS amplitude of the fundamental component in V.
        harmonics_v: RMS amplitudes of harmonic components (2nd, 3rd, ...) in V.

    Returns:
        THD in percent.

    Raises:
        ValueError: If fundamental_v is non-positive.
    """
    if fundamental_v <= 0:
        raise ValueError("fundamental_v must be positive")
    harmonic_sum_sq = sum(h**2 for h in harmonics_v)
    return round(100.0 * math.sqrt(harmonic_sum_sq) / fundamental_v, 4)


def crest_factor(peak_v: float, rms_v: float) -> float:
    """Compute waveform crest factor (peak-to-RMS ratio).

    Args:
        peak_v: Peak voltage in V.
        rms_v: RMS voltage in V.

    Returns:
        Crest factor (dimensionless).

    Raises:
        ValueError: If rms_v is non-positive or peak < rms.
    """
    if rms_v <= 0:
        raise ValueError("rms_v must be positive")
    if peak_v < rms_v:
        raise ValueError("peak_v must be >= rms_v")
    return round(peak_v / rms_v, 4)


def k_factor(
    fundamental_a: float,
    harmonics_a: list[float],
) -> float:
    """Compute K-factor for transformer derating with non-linear loads.

    Args:
        fundamental_a: RMS fundamental current in A.
        harmonics_a: RMS harmonic currents [2nd, 3rd, ...] in A.

    Returns:
        K-factor (dimensionless).

    Raises:
        ValueError: If fundamental_a is non-positive.
    """
    if fundamental_a <= 0:
        raise ValueError("fundamental_a must be positive")
    all_currents = [fundamental_a] + harmonics_a
    i_sq_sum = sum(i**2 for i in all_currents)
    weighted_sum = fundamental_a**2 * 1 + sum(
        (h**2) * ((n + 2) ** 2) for n, h in enumerate(harmonics_a)
    )
    if i_sq_sum == 0:
        return 0.0
    return round(weighted_sum / i_sq_sum, 4)


def classify_voltage_event(
    voltage_v: float,
    nominal_v: float = NOMINAL_VOLTAGE_V,
    sag_threshold_pct: float = 10.0,
    swell_threshold_pct: float = 10.0,
    interruption_threshold_pct: float = 90.0,
) -> PQEvent:
    """Classify a measured voltage as a PQ event type.

    Args:
        voltage_v: Measured RMS voltage in V.
        nominal_v: Nominal voltage in V.
        sag_threshold_pct: Voltage drop % below nominal to classify as sag.
        swell_threshold_pct: Voltage rise % above nominal to classify as swell.
        interruption_threshold_pct: Drop % to classify as interruption.

    Returns:
        Classified :class:`PQEvent`.
    """
    deviation_pct = (voltage_v - nominal_v) / nominal_v * 100.0
    if deviation_pct < -interruption_threshold_pct:
        return PQEvent.INTERRUPTION
    if deviation_pct < -sag_threshold_pct:
        return PQEvent.SAG
    if deviation_pct > swell_threshold_pct:
        return PQEvent.SWELL
    return PQEvent.NORMAL


def en50160_compliance(vrms_v: float, thd_pct: float) -> bool:
    """Check basic EN 50160 compliance for voltage and THD.

    Args:
        vrms_v: Measured RMS voltage in V.
        thd_pct: Measured voltage THD in %.

    Returns:
        True if within EN 50160 limits.
    """
    voltage_ok = 0.90 * NOMINAL_VOLTAGE_V <= vrms_v <= 1.10 * NOMINAL_VOLTAGE_V
    thd_ok = thd_pct <= 8.0
    return voltage_ok and thd_ok


def analyse_power_quality(
    samples: list[float],
    fundamental_v: float,
    harmonics_v: list[float],
    peak_v: float,
    fundamental_a: float,
    harmonics_a: list[float],
) -> PQReport:
    """Run a complete power quality analysis.

    Args:
        samples: Instantaneous voltage samples in V.
        fundamental_v: RMS fundamental voltage in V.
        harmonics_v: RMS harmonic voltages in V.
        peak_v: Peak voltage in V.
        fundamental_a: RMS fundamental current in A.
        harmonics_a: RMS harmonic currents in A.

    Returns:
        Populated :class:`PQReport`.
    """
    vrms = rms_voltage(samples)
    thd = thd_voltage(fundamental_v, harmonics_v)
    cf = crest_factor(peak_v, vrms) if vrms > 0 else 0.0
    kf = k_factor(fundamental_a, harmonics_a)
    event_type = classify_voltage_event(vrms)
    events = []
    if event_type != PQEvent.NORMAL:
        events.append(
            PQEventRecord(
                event_type=event_type,
                magnitude_pct=abs((vrms - NOMINAL_VOLTAGE_V) / NOMINAL_VOLTAGE_V * 100),
                duration_cycles=1.0,
            )
        )
    compliant = en50160_compliance(vrms, thd)
    return PQReport(
        vrms_v=vrms,
        thd_pct=thd,
        crest_factor=cf,
        k_factor=kf,
        events=events,
        within_en50160=compliant,
    )
