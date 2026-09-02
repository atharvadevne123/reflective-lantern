"""Demand-response event evaluation.

Utilities for grid demand-response programmes: establishing a customer
baseline load (CBL), measuring realised curtailment against it, valuing the
event, and scoring how reliably a site delivers what it commits to.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_CBL_DAYS: int = 5
"""Number of comparable historical days averaged into the baseline."""

DEFAULT_INCENTIVE_PER_KWH: float = 1.25
"""Incentive paid per kWh of verified curtailment."""

DEFAULT_PENALTY_PER_KWH: float = 0.75
"""Penalty charged per kWh of committed-but-undelivered curtailment."""


@dataclass
class DemandResponseResult:
    """Outcome of a single demand-response event."""

    baseline_kwh: float
    actual_kwh: float
    curtailed_kwh: float
    curtailment_pct: float
    committed_kwh: float
    shortfall_kwh: float
    incentive: float
    penalty: float
    net_payment: float
    performance_score: float


def customer_baseline_load(
    historical_hourly_kwh: list[list[float]],
    days: int = DEFAULT_CBL_DAYS,
) -> list[float]:
    """Average comparable historical days into an hourly baseline.

    Args:
        historical_hourly_kwh: One list of hourly readings per historical day.
            All days must have the same number of hours.
        days: Number of most-recent days to average.

    Returns:
        Hourly baseline consumption in kWh, rounded to 4 decimal places.

    Raises:
        ValueError: If *historical_hourly_kwh* is empty, *days* is not
            positive, or the days have differing lengths.
    """
    if not historical_hourly_kwh:
        raise ValueError("historical_hourly_kwh must not be empty")
    if days <= 0:
        raise ValueError(f"days must be positive, got {days}")

    window = historical_hourly_kwh[-days:]
    hours = len(window[0])
    for index, day in enumerate(window):
        if len(day) != hours:
            raise ValueError(
                f"all historical days must have the same length; day {index} has {len(day)}, expected {hours}"
            )

    baseline = [round(statistics.fmean(day[hour] for day in window), 4) for hour in range(hours)]
    logger.debug("Baseline built from %d day(s) across %d hour(s)", len(window), hours)
    return baseline


def curtailment(baseline_hourly_kwh: list[float], actual_hourly_kwh: list[float]) -> float:
    """Return kWh curtailed against the baseline over the event window.

    Args:
        baseline_hourly_kwh: Expected hourly consumption absent the event.
        actual_hourly_kwh: Measured hourly consumption during the event.

    Returns:
        Total curtailment in kWh rounded to 4 decimal places. Negative when
        the site consumed more than its baseline.

    Raises:
        ValueError: If the two series differ in length.
    """
    if len(baseline_hourly_kwh) != len(actual_hourly_kwh):
        raise ValueError(
            f"baseline and actual must be the same length (got {len(baseline_hourly_kwh)} vs {len(actual_hourly_kwh)})"
        )
    return round(sum(baseline_hourly_kwh) - sum(actual_hourly_kwh), 4)


def performance_score(curtailed_kwh: float, committed_kwh: float) -> float:
    """Score delivery against commitment, capped at 1.0.

    Args:
        curtailed_kwh: Verified curtailment delivered.
        committed_kwh: Curtailment the site committed to.

    Returns:
        Score in 0-1 rounded to 4 decimal places. Returns ``1.0`` when
        nothing was committed, and ``0.0`` when curtailment was negative.

    Raises:
        ValueError: If *committed_kwh* is negative.
    """
    if committed_kwh < 0:
        raise ValueError(f"committed_kwh must be non-negative, got {committed_kwh}")
    if committed_kwh == 0:
        return 1.0
    return round(max(0.0, min(1.0, curtailed_kwh / committed_kwh)), 4)


def evaluate_event(
    baseline_hourly_kwh: list[float],
    actual_hourly_kwh: list[float],
    committed_kwh: float,
    incentive_per_kwh: float = DEFAULT_INCENTIVE_PER_KWH,
    penalty_per_kwh: float = DEFAULT_PENALTY_PER_KWH,
) -> DemandResponseResult:
    """Evaluate a demand-response event end to end.

    Curtailment is paid at *incentive_per_kwh*; any shortfall against
    *committed_kwh* is charged at *penalty_per_kwh*. Over-delivery is paid
    in full but does not offset a shortfall, since there is none.

    Args:
        baseline_hourly_kwh: Expected hourly consumption absent the event.
        actual_hourly_kwh: Measured hourly consumption during the event.
        committed_kwh: Curtailment the site committed to.
        incentive_per_kwh: Payment per kWh of verified curtailment.
        penalty_per_kwh: Charge per kWh of undelivered commitment.

    Returns:
        A populated :class:`DemandResponseResult`.

    Raises:
        ValueError: If the series differ in length, *committed_kwh* is
            negative, or either rate is negative.
    """
    if incentive_per_kwh < 0 or penalty_per_kwh < 0:
        raise ValueError(f"rates must be non-negative, got incentive={incentive_per_kwh} penalty={penalty_per_kwh}")

    curtailed = curtailment(baseline_hourly_kwh, actual_hourly_kwh)
    baseline_total = round(sum(baseline_hourly_kwh), 4)
    actual_total = round(sum(actual_hourly_kwh), 4)
    curtailment_pct = round(100.0 * curtailed / baseline_total, 2) if baseline_total > 0 else 0.0

    score = performance_score(curtailed, committed_kwh)
    shortfall = round(max(0.0, committed_kwh - curtailed), 4)
    incentive = round(max(0.0, curtailed) * incentive_per_kwh, 2)
    penalty = round(shortfall * penalty_per_kwh, 2)

    logger.info(
        "DR event: curtailed %.2f kWh (%.1f%%), score %.2f, net %.2f",
        curtailed,
        curtailment_pct,
        score,
        incentive - penalty,
    )
    return DemandResponseResult(
        baseline_kwh=baseline_total,
        actual_kwh=actual_total,
        curtailed_kwh=curtailed,
        curtailment_pct=curtailment_pct,
        committed_kwh=round(committed_kwh, 4),
        shortfall_kwh=shortfall,
        incentive=incentive,
        penalty=penalty,
        net_payment=round(incentive - penalty, 2),
        performance_score=score,
    )


__all__ = [
    "DEFAULT_CBL_DAYS",
    "DEFAULT_INCENTIVE_PER_KWH",
    "DEFAULT_PENALTY_PER_KWH",
    "DemandResponseResult",
    "curtailment",
    "customer_baseline_load",
    "evaluate_event",
    "performance_score",
]
