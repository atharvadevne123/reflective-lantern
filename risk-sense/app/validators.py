"""Input validation helpers for Risk-Sense credit applications."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

VALID_RISK_CATEGORIES = frozenset({"low", "medium", "high", "very_high"})


def validate_dti(value: float) -> float:
    """Validate debt-to-income ratio is in [0, 10].

    Args:
        value: Debt-to-income ratio to validate.

    Returns:
        The validated value.

    Raises:
        ValueError: If value is outside [0, 10].
    """
    if not (0.0 <= value <= 10.0):
        raise ValueError(f"debt_to_income must be in [0, 10], got {value}")
    return value


def validate_probability_field(value: float, field_name: str) -> float:
    """Validate a field that must be a probability in [0, 1].

    Args:
        value: Value to validate.
        field_name: Name of the field for error messages.

    Returns:
        The validated value.

    Raises:
        ValueError: If value is outside [0, 1].
    """
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{field_name} must be in [0, 1], got {value}")
    return value


def validate_positive_integer(value: int, field_name: str) -> int:
    """Validate a non-negative integer field.

    Args:
        value: Integer to validate.
        field_name: Name of the field for error messages.

    Returns:
        The validated value.

    Raises:
        ValueError: If value is negative.
    """
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value}")
    return value


def validate_income(value: float) -> float:
    """Validate annual income is non-negative.

    Args:
        value: Annual income to validate.

    Returns:
        The validated value.

    Raises:
        ValueError: If value is negative.
    """
    if value < 0:
        raise ValueError(f"annual_income must be non-negative, got {value}")
    return value


def validate_application(data: dict[str, Any]) -> dict[str, Any]:
    """Run all field-level validation on a raw credit application dict.

    Args:
        data: Raw application dictionary with credit fields.

    Returns:
        Validated and coerced application dict.

    Raises:
        ValueError: If any field fails validation.
    """
    errors: list[str] = []
    validated = dict(data)

    try:
        validated["debt_to_income"] = validate_dti(float(data.get("debt_to_income", 0)))
    except (ValueError, TypeError) as e:
        errors.append(str(e))

    for field in ("payment_history_score", "credit_utilization"):
        try:
            validated[field] = validate_probability_field(float(data.get(field, 0)), field)
        except (ValueError, TypeError) as e:
            errors.append(str(e))

    for field in ("account_age_months", "num_open_accounts", "num_derogatory_marks"):
        try:
            validated[field] = validate_positive_integer(int(data.get(field, 0)), field)
        except (ValueError, TypeError) as e:
            errors.append(str(e))

    try:
        validated["annual_income"] = validate_income(float(data.get("annual_income", 0)))
    except (ValueError, TypeError) as e:
        errors.append(str(e))

    if errors:
        logger.warning("Validation errors: %s", errors)
        raise ValueError(f"Validation failed: {'; '.join(errors)}")

    return validated
