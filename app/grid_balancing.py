"""Grid frequency regulation and ancillary services modelling."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

NOMINAL_FREQUENCY_HZ: float = 50.0
"""Nominal grid frequency in Hz (European standard)."""

NOMINAL_FREQUENCY_HZ_US: float = 60.0
"""Nominal grid frequency in Hz (North American standard)."""

PRIMARY_RESPONSE_DEADBAND_HZ: float = 0.015
"""Typical primary response activation deadband in Hz."""

SECONDARY_RESPONSE_DEADBAND_HZ: float = 0.10
"""Secondary / droop activation threshold in Hz."""

DROOP_PERCENT_DEFAULT: float = 4.0
"""Default governor droop setting as a percentage."""


@dataclass
class FrequencyEvent:
    """Characterisation of a grid frequency deviation event."""

    nadir_hz: float
    rocof_hz_per_s: float
    steady_state_deviation_hz: float
    nominal_hz: float
    primary_response_required: bool
    secondary_response_required: bool
    severity: str


@dataclass
class AncillaryServiceBid:
    """Bid/offer for a frequency response ancillary service."""

    asset_id: str
    service_type: str
    capacity_mw: float
    response_time_s: float
    duration_min: float
    price_per_mw: float
    total_value: float


def frequency_deviation(
    measured_hz: float,
    nominal_hz: float = NOMINAL_FREQUENCY_HZ,
) -> float:
    """Compute frequency deviation from nominal.

    Args:
        measured_hz: Measured frequency in Hz.
        nominal_hz: Nominal frequency in Hz.

    Returns:
        Signed deviation in Hz (positive = over-frequency).

    Raises:
        ValueError: If nominal_hz <= 0.
    """
    if nominal_hz <= 0:
        raise ValueError("nominal_hz must be positive")
    return round(measured_hz - nominal_hz, 6)


def rocof(
    freq_samples_hz: list[float],
    sample_interval_s: float,
) -> float:
    """Estimate Rate of Change of Frequency (RoCoF) from a sample sequence.

    Uses a linear least-squares fit over the window.

    Args:
        freq_samples_hz: Sequence of frequency measurements in Hz.
        sample_interval_s: Time between samples in seconds.

    Returns:
        RoCoF in Hz/s (negative = declining frequency).

    Raises:
        ValueError: If fewer than 2 samples or interval <= 0.
    """
    if len(freq_samples_hz) < 2:
        raise ValueError("At least 2 samples required")
    if sample_interval_s <= 0:
        raise ValueError("sample_interval_s must be positive")
    n = len(freq_samples_hz)
    times = [i * sample_interval_s for i in range(n)]
    mean_t = sum(times) / n
    mean_f = sum(freq_samples_hz) / n
    num = sum((t - mean_t) * (f - mean_f) for t, f in zip(times, freq_samples_hz))
    den = sum((t - mean_t) ** 2 for t in times)
    if den == 0:
        return 0.0
    return round(num / den, 6)


def droop_response_kw(
    frequency_deviation_hz: float,
    rated_capacity_kw: float,
    nominal_hz: float = NOMINAL_FREQUENCY_HZ,
    droop_percent: float = DROOP_PERCENT_DEFAULT,
) -> float:
    """Power response from a generator with governor droop control.

    Args:
        frequency_deviation_hz: Frequency deviation in Hz (negative = under-frequency).
        rated_capacity_kw: Unit rated capacity in kW.
        nominal_hz: Nominal grid frequency in Hz.
        droop_percent: Governor droop setting as a percentage of nominal.

    Returns:
        Power response in kW (positive = inject power, negative = absorb).

    Raises:
        ValueError: If rated capacity <= 0 or droop <= 0.
    """
    if rated_capacity_kw <= 0:
        raise ValueError("rated_capacity_kw must be positive")
    if droop_percent <= 0:
        raise ValueError("droop_percent must be positive")
    droop_fraction = droop_percent / 100.0
    response_fraction = -frequency_deviation_hz / (droop_fraction * nominal_hz)
    response_kw = response_fraction * rated_capacity_kw
    return round(max(-rated_capacity_kw, min(rated_capacity_kw, response_kw)), 4)


def inertia_constant(
    stored_energy_mj: float,
    rated_power_mva: float,
) -> float:
    """Compute the inertia constant H of a synchronous machine.

    Args:
        stored_energy_mj: Kinetic energy stored at rated speed in MJ.
        rated_power_mva: Machine rated apparent power in MVA.

    Returns:
        Inertia constant H in seconds (MWs/MVA).

    Raises:
        ValueError: If either value is non-positive.
    """
    if stored_energy_mj <= 0:
        raise ValueError("stored_energy_mj must be positive")
    if rated_power_mva <= 0:
        raise ValueError("rated_power_mva must be positive")
    return round(stored_energy_mj / rated_power_mva, 4)


def swing_equation_rocof(
    power_imbalance_mw: float,
    system_inertia_mws: float,
    nominal_hz: float = NOMINAL_FREQUENCY_HZ,
) -> float:
    """Initial RoCoF from the swing equation: df/dt = f0/(2H) * (Pm - Pe)/S.

    Args:
        power_imbalance_mw: Generation minus load in MW (negative = deficit).
        system_inertia_mws: System inertia in MWs.
        nominal_hz: Nominal frequency in Hz.

    Returns:
        Predicted RoCoF in Hz/s.

    Raises:
        ValueError: If system_inertia_mws <= 0.
    """
    if system_inertia_mws <= 0:
        raise ValueError("system_inertia_mws must be positive")
    return round(nominal_hz * power_imbalance_mw / (2.0 * system_inertia_mws), 6)


def classify_frequency_event(
    nadir_hz: float,
    rocof_hz_per_s: float,
    nominal_hz: float = NOMINAL_FREQUENCY_HZ,
) -> FrequencyEvent:
    """Classify a frequency event by severity.

    Args:
        nadir_hz: Frequency nadir (lowest point) in Hz.
        rocof_hz_per_s: Observed RoCoF in Hz/s.
        nominal_hz: Nominal frequency in Hz.

    Returns:
        Populated :class:`FrequencyEvent`.
    """
    steady_dev = frequency_deviation(nadir_hz, nominal_hz)
    primary = abs(steady_dev) > PRIMARY_RESPONSE_DEADBAND_HZ
    secondary = abs(steady_dev) > SECONDARY_RESPONSE_DEADBAND_HZ
    if abs(steady_dev) < 0.1 and abs(rocof_hz_per_s) < 0.5:
        severity = "normal"
    elif abs(steady_dev) < 0.5 and abs(rocof_hz_per_s) < 1.0:
        severity = "minor"
    elif abs(steady_dev) < 1.0 or abs(rocof_hz_per_s) < 2.0:
        severity = "moderate"
    else:
        severity = "severe"

    return FrequencyEvent(
        nadir_hz=nadir_hz,
        rocof_hz_per_s=rocof_hz_per_s,
        steady_state_deviation_hz=steady_dev,
        nominal_hz=nominal_hz,
        primary_response_required=primary,
        secondary_response_required=secondary,
        severity=severity,
    )


def ancillary_service_value(
    asset_id: str,
    capacity_mw: float,
    service_type: str,
    response_time_s: float,
    duration_min: float,
    price_per_mw: float,
) -> AncillaryServiceBid:
    """Compute the value of an ancillary service bid.

    Args:
        asset_id: Asset identifier.
        capacity_mw: Offered capacity in MW.
        service_type: Service name, e.g. ``'FFR'``, ``'PFR'``, ``'SFR'``.
        response_time_s: Response time in seconds.
        duration_min: Sustained duration in minutes.
        price_per_mw: Market clearing price in $/MW.

    Returns:
        Populated :class:`AncillaryServiceBid`.

    Raises:
        ValueError: If capacity or price are non-positive.
    """
    if capacity_mw <= 0:
        raise ValueError("capacity_mw must be positive")
    if price_per_mw < 0:
        raise ValueError("price_per_mw must be non-negative")
    total = capacity_mw * price_per_mw
    return AncillaryServiceBid(
        asset_id=asset_id,
        service_type=service_type,
        capacity_mw=round(capacity_mw, 4),
        response_time_s=response_time_s,
        duration_min=duration_min,
        price_per_mw=price_per_mw,
        total_value=round(total, 4),
    )
