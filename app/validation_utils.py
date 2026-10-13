"""Lightweight validation helpers shared across Watt-Guard modules."""

from __future__ import annotations

__all__ = [
    "is_valid_email",
    "is_valid_percentage",
    "is_valid_temperature_c",
    "is_valid_url",
    "validate_positive",
    "validate_range",
]


def validate_positive(value: float, name: str = "value") -> float:
    """Assert *value* is strictly positive and return it.

    Args:
        value: The numeric value to validate.
        name: Human-readable name used in the error message.

    Returns:
        *value* unchanged when valid.

    Raises:
        ValueError: If *value* is not strictly positive (> 0).
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def validate_range(value: float, low: float, high: float, name: str = "value") -> float:
    """Assert *value* is within [*low*, *high*] and return it.

    Args:
        value: The numeric value to validate.
        low: Inclusive lower bound.
        high: Inclusive upper bound.
        name: Human-readable name used in the error message.

    Returns:
        *value* unchanged when valid.

    Raises:
        ValueError: If *value* is outside the specified range.
    """
    if not (low <= value <= high):
        raise ValueError(f"{name} must be in [{low}, {high}], got {value}")
    return value


def is_valid_percentage(value: float) -> bool:
    """Return True if *value* is a valid percentage in [0.0, 100.0].

    Args:
        value: The float to check.

    Returns:
        True when 0.0 <= value <= 100.0.
    """
    return 0.0 <= value <= 100.0


def is_valid_temperature_c(value: float) -> bool:
    """Return True if *value* is a plausible Celsius temperature for buildings.

    Accepts the range [-50, 60] C as defined in constants.

    Args:
        value: Temperature in degrees Celsius.

    Returns:
        True when within the valid range.
    """
    return -50.0 <= value <= 60.0


def is_valid_email(email: str) -> bool:
    """Return True if *email* passes a basic structural check.

    Checks for exactly one ``@``, a non-empty local part, and at least one
    dot in the domain.  Does not perform DNS resolution.

    Args:
        email: Email address string.

    Returns:
        True for structurally plausible email addresses.
    """
    parts = email.split("@")
    if len(parts) != 2:
        return False
    local, domain = parts
    if not local or not domain:
        return False
    return "." in domain and not domain.startswith(".") and not domain.endswith(".")


def is_valid_url(url: str) -> bool:
    """Return True if *url* starts with http:// or https:// and has a host.

    Args:
        url: URL string to validate.

    Returns:
        True for structurally plausible HTTP/HTTPS URLs.
    """
    for scheme in ("http://", "https://"):
        if url.startswith(scheme):
            remainder = url[len(scheme):]
            return bool(remainder) and not remainder.startswith("/")
    return False
