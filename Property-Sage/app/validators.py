"""Shared Pydantic validators and input sanitisation for Property-Sage.

Central home for reusable validators so endpoint schemas stay lean.
"""

from __future__ import annotations

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
