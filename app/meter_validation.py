"""Smart meter data validation and gap-detection utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

logger = logging.getLogger(__name__)

MAX_PLAUSIBLE_KWH_PER_INTERVAL: float = 500.0
"""Upper bound for a single 30-min meter reading (kWh)."""

MIN_PLAUSIBLE_KWH: float = 0.0
"""Readings below this value are physically impossible."""


class ValidationStatus(StrEnum):
    """Overall validation result for a meter reading series."""

    VALID = "valid"
    WARNINGS = "warnings"
    INVALID = "invalid"


@dataclass
class ValidationIssue:
    """A single data quality issue found in a meter series."""

    index: int
    kind: str
    description: str
    value: float | None = None


@dataclass
class ValidationReport:
    """Full validation report for a meter reading series."""

    status: ValidationStatus
    total_readings: int
    issues: list[ValidationIssue] = field(default_factory=list)
    missing_intervals: list[int] = field(default_factory=list)
    gap_count: int = 0
    outlier_count: int = 0


def check_negatives(readings: list[float]) -> list[ValidationIssue]:
    """Flag any negative readings as invalid.

    Args:
        readings: Meter readings in kWh.

    Returns:
        List of :class:`ValidationIssue` for each negative value.
    """
    return [
        ValidationIssue(i, "negative_reading", f"Negative value {v}", v)
        for i, v in enumerate(readings)
        if v < MIN_PLAUSIBLE_KWH
    ]


def check_upper_bound(
    readings: list[float],
    upper: float = MAX_PLAUSIBLE_KWH_PER_INTERVAL,
) -> list[ValidationIssue]:
    """Flag readings that exceed a physical plausibility threshold.

    Args:
        readings: Meter readings in kWh.
        upper: Maximum plausible reading per interval.

    Returns:
        List of :class:`ValidationIssue` for values above ``upper``.
    """
    return [
        ValidationIssue(i, "exceeds_upper_bound", f"Value {v} exceeds {upper}", v)
        for i, v in enumerate(readings)
        if v > upper
    ]


def check_zeros(
    readings: list[float],
    max_consecutive_zeros: int = 4,
) -> list[ValidationIssue]:
    """Flag runs of consecutive zero readings beyond a configurable limit.

    Args:
        readings: Meter readings in kWh.
        max_consecutive_zeros: Maximum allowed consecutive zeros before flagging.

    Returns:
        :class:`ValidationIssue` for each position inside a too-long zero run.
    """
    issues: list[ValidationIssue] = []
    run_start: int | None = None
    run_len = 0
    for i, v in enumerate(readings):
        if v == 0.0:
            if run_start is None:
                run_start = i
            run_len += 1
        else:
            if run_len > max_consecutive_zeros and run_start is not None:
                for j in range(run_start, run_start + run_len):
                    issues.append(
                        ValidationIssue(
                            j,
                            "excessive_zeros",
                            f"Zero run of {run_len} starting at {run_start}",
                        )
                    )
            run_start = None
            run_len = 0
    if run_len > max_consecutive_zeros and run_start is not None:
        for j in range(run_start, run_start + run_len):
            issues.append(
                ValidationIssue(
                    j,
                    "excessive_zeros",
                    f"Zero run of {run_len} starting at {run_start}",
                )
            )
    return issues


def detect_gaps(
    timestamps: list[int],
    expected_interval_s: int = 1800,
) -> list[int]:
    """Find indices where the gap between consecutive timestamps is too large.

    Args:
        timestamps: UNIX timestamps in ascending order (seconds).
        expected_interval_s: Expected spacing between readings in seconds.

    Returns:
        List of indices (0-based) where the gap before that reading exceeds
        1.5× the expected interval.

    Raises:
        ValueError: If timestamps is not sorted in ascending order.
    """
    if len(timestamps) < 2:
        return []
    threshold = expected_interval_s * 1.5
    gaps: list[int] = []
    for i in range(1, len(timestamps)):
        if timestamps[i] < timestamps[i - 1]:
            raise ValueError(f"Timestamps not sorted at index {i}")
        if (timestamps[i] - timestamps[i - 1]) > threshold:
            gaps.append(i)
    return gaps


def validate_readings(
    readings: list[float],
    upper: float = MAX_PLAUSIBLE_KWH_PER_INTERVAL,
    max_zeros: int = 4,
) -> ValidationReport:
    """Run all validation checks on a meter reading series.

    Args:
        readings: Meter readings in kWh.
        upper: Per-interval plausibility ceiling.
        max_zeros: Maximum allowed consecutive zero readings.

    Returns:
        :class:`ValidationReport` summarising all issues found.
    """
    if not readings:
        return ValidationReport(
            status=ValidationStatus.INVALID,
            total_readings=0,
            issues=[ValidationIssue(0, "empty_series", "No readings provided")],
        )
    issues: list[ValidationIssue] = []
    issues.extend(check_negatives(readings))
    issues.extend(check_upper_bound(readings, upper))
    issues.extend(check_zeros(readings, max_zeros))
    outlier_count = sum(
        1 for iss in issues if iss.kind in ("negative_reading", "exceeds_upper_bound")
    )
    if any(iss.kind == "negative_reading" for iss in issues):
        status = ValidationStatus.INVALID
    elif issues:
        status = ValidationStatus.WARNINGS
    else:
        status = ValidationStatus.VALID
    return ValidationReport(
        status=status,
        total_readings=len(readings),
        issues=issues,
        outlier_count=outlier_count,
    )
