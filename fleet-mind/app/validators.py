"""Business-logic validation helpers for fleet telemetry readings."""

from __future__ import annotations

import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Physical plausibility bounds beyond Pydantic field constraints
_PLAUSIBILITY_RULES: list[tuple[str, float, float, str]] = [
    ("engine_temp_c", 50.0, 160.0, "Engine temperature outside safe operating range"),
    ("oil_pressure_bar", 0.3, 8.0, "Oil pressure outside safe operating range"),
    ("brake_pad_mm", 0.0, 15.0, "Brake pad thickness outside expected range"),
    ("battery_voltage_v", 9.0, 16.0, "Battery voltage outside expected range"),
    ("tire_pressure_bar", 0.5, 4.5, "Tyre pressure outside expected range"),
    ("rpm", 0.0, 8000.0, "RPM outside expected range"),
    ("coolant_temp_c", 40.0, 130.0, "Coolant temperature outside safe operating range"),
    ("transmission_temp_c", 40.0, 170.0, "Transmission temperature outside safe operating range"),
]


class ValidationIssue(NamedTuple):
    """A single validation finding."""

    field: str
    message: str
    severity: str  # 'warning' | 'error'


def validate_reading(reading: dict[str, float]) -> list[ValidationIssue]:
    """Run plausibility checks on a telemetry reading dict.

    Args:
        reading: Mapping from feature name to float value.

    Returns:
        List of ValidationIssue objects. Empty list means all checks passed.
    """
    issues: list[ValidationIssue] = []

    for field, low, high, message in _PLAUSIBILITY_RULES:
        value = reading.get(field)
        if value is None:
            continue
        if not (low <= value <= high):
            severity = "error" if (value < low * 0.5 or value > high * 1.5) else "warning"
            issues.append(ValidationIssue(field=field, message=message, severity=severity))
            logger.warning("Plausibility check failed for %s=%.2f: %s", field, value, message)

    return issues


def has_critical_issues(issues: list[ValidationIssue]) -> bool:
    """Return True if any issue has severity 'error'.

    Args:
        issues: Output from validate_reading().

    Returns:
        True when at least one error-severity issue is present.
    """
    return any(i.severity == "error" for i in issues)
