"""
Input validation for Price-Prophet data records.

All validators return a :class:`ValidationResult` so callers can
decide how to handle failures rather than having exceptions thrown
unconditionally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Outcome of a validation check.

    Attributes
    ----------
    valid:
        ``True`` if no errors were found.
    errors:
        Human-readable descriptions of every problem detected.
    """

    valid: bool
    errors: list[str] = field(default_factory=list)


# Required keys that every pricing record must have
_REQUIRED_RECORD_KEYS = {"product_id", "base_price", "demand"}


def validate_price(price: float) -> ValidationResult:
    """Check that *price* is a strictly positive number.

    Parameters
    ----------
    price:
        Numeric price value to validate.

    Returns
    -------
    ValidationResult
    """
    errors: list[str] = []

    try:
        price = float(price)
    except (TypeError, ValueError):
        errors.append(f"price must be numeric, got {type(price).__name__!r}")
        return ValidationResult(valid=False, errors=errors)

    if price <= 0.0:
        errors.append(f"price must be > 0, got {price}")

    return ValidationResult(valid=len(errors) == 0, errors=errors)


def validate_record(record: dict[str, Any]) -> ValidationResult:
    """Validate a single pricing record dict.

    Checks
    ------
    * All required keys (``product_id``, ``base_price``, ``demand``)
      are present.
    * ``base_price`` is strictly positive.
    * ``demand`` is non-negative.

    Parameters
    ----------
    record:
        Raw record dict as returned by a loader function.

    Returns
    -------
    ValidationResult
    """
    errors: list[str] = []

    missing = _REQUIRED_RECORD_KEYS - set(record.keys())
    if missing:
        errors.append(f"missing required keys: {sorted(missing)}")
        return ValidationResult(valid=False, errors=errors)

    # Validate base_price
    try:
        base_price = float(record["base_price"])
        if base_price <= 0.0:
            errors.append(
                f"base_price must be > 0, got {base_price} (product_id={record['product_id']!r})"
            )
    except (TypeError, ValueError):
        errors.append(
            f"base_price must be numeric, got "
            f"{type(record['base_price']).__name__!r} "
            f"(product_id={record['product_id']!r})"
        )

    # Validate demand
    try:
        demand = float(record["demand"])
        if demand < 0.0:
            errors.append(
                f"demand must be >= 0, got {demand} (product_id={record['product_id']!r})"
            )
    except (TypeError, ValueError):
        errors.append(
            f"demand must be numeric, got "
            f"{type(record['demand']).__name__!r} "
            f"(product_id={record['product_id']!r})"
        )

    return ValidationResult(valid=len(errors) == 0, errors=errors)


def validate_dataset(records: list[dict[str, Any]]) -> ValidationResult:
    """Validate an entire dataset of pricing records.

    The dataset is considered invalid if it is empty, or if any
    individual record fails validation.

    Parameters
    ----------
    records:
        List of raw record dicts.

    Returns
    -------
    ValidationResult
        Aggregated result; ``errors`` contains one entry per failing
        record (prefixed with its position in the list).
    """
    all_errors: list[str] = []

    if not records:
        logger.debug("validate_dataset: empty dataset")
        return ValidationResult(valid=False, errors=["dataset is empty"])

    for idx, record in enumerate(records):
        result = validate_record(record)
        if not result.valid:
            for err in result.errors:
                all_errors.append(f"record[{idx}]: {err}")

    result = ValidationResult(valid=len(all_errors) == 0, errors=all_errors)
    if not result.valid:
        logger.debug("validate_dataset: %d error(s) in %d records", len(all_errors), len(records))
    return result
