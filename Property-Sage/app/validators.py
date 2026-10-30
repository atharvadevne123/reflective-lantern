"""Shared Pydantic validators and input sanitisation for Property-Sage.

Central home for reusable validators so endpoint schemas stay lean.
"""

from __future__ import annotations

import re

from app.features import NEIGHBORHOODS, PROPERTY_TYPES


def validate_neighborhood_value(v: str) -> str:
    """Normalise and validate a neighbourhood string.

    Args:
        v: Raw neighbourhood value from the request.

    Returns:
        Lowercased, stripped neighbourhood string.

    Raises:
        ValueError: If the value is not in the known neighbourhood list.
    """
    normalised = v.lower().strip()
    if normalised not in NEIGHBORHOODS:
        raise ValueError(f"neighbourhood must be one of: {', '.join(sorted(NEIGHBORHOODS))}")
    return normalised


def validate_property_type_value(v: str) -> str:
    """Normalise and validate a property type string.

    Args:
        v: Raw property_type value from the request.

    Returns:
        Lowercased, stripped property type string.

    Raises:
        ValueError: If the value is not in the known property type list.
    """
    normalised = v.lower().strip()
    if normalised not in PROPERTY_TYPES:
        raise ValueError(f"property_type must be one of: {', '.join(sorted(PROPERTY_TYPES))}")
    return normalised


def validate_sqft_vs_bedrooms(sqft: float, bedrooms: int) -> None:
    """Raise if the sqft-per-bedroom ratio is implausibly low.

    Args:
        sqft: Total interior square footage.
        bedrooms: Number of bedrooms.

    Raises:
        ValueError: If sqft per bedroom is below 80 sqft.
    """
    ratio = sqft / max(bedrooms, 1)
    if ratio < 80:
        raise ValueError(
            f"sqft_per_bedroom={ratio:.1f} is implausibly low "
            f"(minimum 80 sqft per bedroom expected)"
        )


def validate_year_built_value(v: int) -> int:
    """Validate that year_built is a plausible construction year.

    Args:
        v: Year built value from the request.

    Returns:
        The validated year.

    Raises:
        ValueError: If the year is before 1800 or more than 1 year in the future.
    """
    import datetime
    current_year = datetime.date.today().year
    if v < 1800:
        raise ValueError("year_built must be 1800 or later")
    if v > current_year + 1:
        raise ValueError(f"year_built cannot be in the future (got {v})")
    return v


def normalize_neighborhood(v: str) -> str:
    """Normalise a neighbourhood string without raising on unknown values.

    Args:
        v: Raw neighbourhood string from any source.

    Returns:
        Lowercased, stripped string. Returns 'unknown' if empty after stripping.
    """
    normalised = v.lower().strip() if v else ""
    return normalised if normalised else "unknown"


def sanitize_request_id(request_id: str) -> str:
    """Sanitise a request ID to contain only safe alphanumeric characters.

    Args:
        request_id: Raw request ID string.

    Returns:
        Sanitised string with only alphanumeric chars and hyphens, max 64 chars.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9\-]", "", request_id)
    return sanitized[:64]
