"""Renewable energy curtailment analysis and lost revenue estimation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

GRID_EMISSION_FACTOR_KG_CO2_PER_KWH: float = 0.233
"""Average grid CO2 intensity in kg/kWh."""


@dataclass
class CurtailmentEvent:
    """A single curtailment event."""

    start_interval: int
    end_interval: int
    available_kw: float
    curtailed_kw: float
    duration_hours: float
    energy_curtailed_kwh: float
    revenue_lost: float


@dataclass
class CurtailmentSummary:
    """Aggregated curtailment statistics for a period."""

    total_available_kwh: float
    total_generated_kwh: float
    total_curtailed_kwh: float
    curtailment_rate_pct: float
    total_revenue_lost: float
    co2_impact_kg: float
    num_events: int
    events: list[CurtailmentEvent] = field(default_factory=list)


def curtailed_energy(
    available_kw: float,
    dispatched_kw: float,
    duration_hours: float,
) -> float:
    """Energy curtailed in a single dispatch interval.

    Args:
        available_kw: Available generation capacity in kW.
        dispatched_kw: Actual dispatched power in kW.
        duration_hours: Interval duration in hours.

    Returns:
        Curtailed energy in kWh (>= 0).

    Raises:
        ValueError: If available_kw < 0, dispatched_kw < 0, or duration <= 0.
    """
    if available_kw < 0:
        raise ValueError("available_kw must be non-negative")
    if dispatched_kw < 0:
        raise ValueError("dispatched_kw must be non-negative")
    if duration_hours <= 0:
        raise ValueError("duration_hours must be positive")
    curtailment_kw = max(0.0, available_kw - dispatched_kw)
    return round(curtailment_kw * duration_hours, 6)


def curtailment_rate(
    total_available_kwh: float,
    total_curtailed_kwh: float,
) -> float:
    """Overall curtailment rate as a percentage.

    Args:
        total_available_kwh: Total available energy over the period in kWh.
        total_curtailed_kwh: Total curtailed energy in kWh.

    Returns:
        Curtailment rate as a percentage.

    Raises:
        ValueError: If curtailed exceeds available or either is negative.
    """
    if total_available_kwh < 0:
        raise ValueError("total_available_kwh must be non-negative")
    if total_curtailed_kwh < 0:
        raise ValueError("total_curtailed_kwh must be non-negative")
    if total_curtailed_kwh > total_available_kwh:
        raise ValueError("curtailed energy cannot exceed available energy")
    if total_available_kwh == 0:
        return 0.0
    return round(100.0 * total_curtailed_kwh / total_available_kwh, 4)


def revenue_lost(
    curtailed_kwh: float,
    energy_price_per_kwh: float,
) -> float:
    """Revenue lost due to curtailment.

    Args:
        curtailed_kwh: Energy curtailed in kWh.
        energy_price_per_kwh: Assumed market price in $/kWh.

    Returns:
        Revenue lost in dollars.

    Raises:
        ValueError: If curtailed_kwh < 0 or price < 0.
    """
    if curtailed_kwh < 0:
        raise ValueError("curtailed_kwh must be non-negative")
    if energy_price_per_kwh < 0:
        raise ValueError("energy_price_per_kwh must be non-negative")
    return round(curtailed_kwh * energy_price_per_kwh, 4)


def analyse_curtailment(
    available_kw_series: list[float],
    dispatched_kw_series: list[float],
    interval_hours: float = 0.5,
    energy_price_per_kwh: float = 0.05,
    co2_factor: float = GRID_EMISSION_FACTOR_KG_CO2_PER_KWH,
) -> CurtailmentSummary:
    """Analyse curtailment over a generation profile.

    Args:
        available_kw_series: Available power at each interval in kW.
        dispatched_kw_series: Dispatched power at each interval in kW.
        interval_hours: Duration of each interval in hours.
        energy_price_per_kwh: Market price for lost revenue calculation in $/kWh.
        co2_factor: CO2 emission factor in kg/kWh for opportunity cost.

    Returns:
        Populated :class:`CurtailmentSummary`.

    Raises:
        ValueError: If series lengths differ or interval_hours <= 0.
    """
    if len(available_kw_series) != len(dispatched_kw_series):
        raise ValueError("available and dispatched series must have the same length")
    if interval_hours <= 0:
        raise ValueError("interval_hours must be positive")
    if not available_kw_series:
        raise ValueError("series must not be empty")

    events: list[CurtailmentEvent] = []
    total_available = 0.0
    total_generated = 0.0
    total_curtailed = 0.0
    total_revenue = 0.0

    in_event = False
    event_start = 0
    event_avail_kw = 0.0
    event_curtailed_kw = 0.0

    for i, (avail, disp) in enumerate(zip(available_kw_series, dispatched_kw_series, strict=False)):
        avail = max(0.0, avail)
        disp = max(0.0, min(avail, disp))
        c_kw = avail - disp
        c_kwh = c_kw * interval_hours
        total_available += avail * interval_hours
        total_generated += disp * interval_hours
        total_curtailed += c_kwh
        total_revenue += revenue_lost(c_kwh, energy_price_per_kwh)

        if c_kw > 0:
            if not in_event:
                in_event = True
                event_start = i
                event_avail_kw = avail
                event_curtailed_kw = c_kw
            else:
                event_avail_kw = max(event_avail_kw, avail)
                event_curtailed_kw += c_kw
        else:
            if in_event:
                dur = (i - event_start) * interval_hours
                e_kwh = event_curtailed_kw * interval_hours
                events.append(
                    CurtailmentEvent(
                        start_interval=event_start,
                        end_interval=i - 1,
                        available_kw=round(event_avail_kw, 4),
                        curtailed_kw=round(event_curtailed_kw, 4),
                        duration_hours=round(dur, 4),
                        energy_curtailed_kwh=round(e_kwh, 4),
                        revenue_lost=round(revenue_lost(e_kwh, energy_price_per_kwh), 4),
                    )
                )
                in_event = False

    if in_event:
        dur = (len(available_kw_series) - event_start) * interval_hours
        e_kwh = event_curtailed_kw * interval_hours
        events.append(
            CurtailmentEvent(
                start_interval=event_start,
                end_interval=len(available_kw_series) - 1,
                available_kw=round(event_avail_kw, 4),
                curtailed_kw=round(event_curtailed_kw, 4),
                duration_hours=round(dur, 4),
                energy_curtailed_kwh=round(e_kwh, 4),
                revenue_lost=round(revenue_lost(e_kwh, energy_price_per_kwh), 4),
            )
        )

    rate = curtailment_rate(total_available, total_curtailed)
    co2_impact = total_curtailed * co2_factor

    return CurtailmentSummary(
        total_available_kwh=round(total_available, 4),
        total_generated_kwh=round(total_generated, 4),
        total_curtailed_kwh=round(total_curtailed, 4),
        curtailment_rate_pct=rate,
        total_revenue_lost=round(total_revenue, 4),
        co2_impact_kg=round(co2_impact, 4),
        num_events=len(events),
        events=events,
    )


def carbon_abatement_lost(
    curtailed_energy_kwh: float,
    carbon_intensity_kg_per_kwh: float = 0.233,
) -> float:
    """Return the CO2 abatement opportunity lost due to curtailment in kg.

    Args:
        curtailed_energy_kwh: Energy curtailed in kWh (non-negative).
        carbon_intensity_kg_per_kwh: Grid carbon intensity in kg CO2e/kWh
            (default 0.233, UK 2024 average).

    Returns:
        Lost CO2 abatement in kg CO2e.

    Raises:
        ValueError: If curtailed_energy_kwh < 0 or carbon_intensity_kg_per_kwh < 0.
    """
    if curtailed_energy_kwh < 0:
        raise ValueError("curtailed_energy_kwh must be non-negative")
    if carbon_intensity_kg_per_kwh < 0:
        raise ValueError("carbon_intensity_kg_per_kwh must be non-negative")
    return curtailed_energy_kwh * carbon_intensity_kg_per_kwh


def curtailment_forecast_accuracy(
    forecast_curtailment_mwh: float,
    actual_curtailment_mwh: float,
) -> float:
    """Return forecast accuracy as a percentage (100% = perfect forecast).

    Calculated as max(0, 1 - |error|/actual) * 100.

    Args:
        forecast_curtailment_mwh: Forecasted curtailment in MWh.
        actual_curtailment_mwh: Actual curtailment in MWh (must be positive).

    Returns:
        Accuracy percentage in [0, 100].

    Raises:
        ValueError: If actual_curtailment_mwh ≤ 0.
    """
    if actual_curtailment_mwh <= 0:
        raise ValueError("actual_curtailment_mwh must be positive")
    error_fraction = abs(forecast_curtailment_mwh - actual_curtailment_mwh) / actual_curtailment_mwh
    return max(0.0, (1.0 - error_fraction) * 100.0)


def value_of_curtailment_reduction(
    curtailment_reduction_kwh: float,
    marginal_generation_cost: float,
    curtailment_payment: float = 0.0,
) -> float:
    """Return the net value of reducing curtailment by a given amount.

    Args:
        curtailment_reduction_kwh: Reduction in curtailed energy in kWh.
        marginal_generation_cost: Marginal cost of curtailed generation per kWh.
        curtailment_payment: Any curtailment payment received per kWh (default 0).

    Returns:
        Net value in currency units (positive means net benefit).

    Raises:
        ValueError: If curtailment_reduction_kwh < 0.
    """
    if curtailment_reduction_kwh < 0:
        raise ValueError("curtailment_reduction_kwh must be non-negative")
    return curtailment_reduction_kwh * (marginal_generation_cost - curtailment_payment)
