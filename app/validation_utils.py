"""Lightweight validation helpers shared across Watt-Guard modules."""

from __future__ import annotations

__all__ = [
    "clamp_to_range",
    "is_non_empty_string",
    "is_valid_email",
    "is_valid_hostname",
    "is_valid_ipv4",
    "is_valid_percentage",
    "is_valid_port",
    "is_valid_temperature_c",
    "is_valid_url",
    "validate_non_negative",
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
            remainder = url[len(scheme) :]
            return bool(remainder) and not remainder.startswith("/")
    return False


def is_non_empty_string(value: str) -> bool:
    """Return True if *value* is a string with at least one non-whitespace character.

    Args:
        value: String to check.

    Returns:
        True when the stripped string is non-empty.
    """
    return isinstance(value, str) and bool(value.strip())


def is_valid_port(port: int) -> bool:
    """Return True if *port* is a valid TCP/UDP port number (1–65535).

    Args:
        port: Integer port number.

    Returns:
        True for values in [1, 65535].
    """
    return isinstance(port, int) and 1 <= port <= 65535


def clamp_to_range(value: float, low: float, high: float) -> float:
    """Return *value* clamped to [*low*, *high*].

    Args:
        value: Numeric value to clamp.
        low: Inclusive lower bound.
        high: Inclusive upper bound.

    Returns:
        *value* bounded to [*low*, *high*].
    """
    return max(low, min(high, value))


def is_valid_ipv4(address: str) -> bool:
    """Return True if *address* is a valid IPv4 address string.

    Args:
        address: String to check, e.g. ``"192.168.1.1"``.

    Returns:
        True for valid dotted-decimal IPv4 addresses; False otherwise.
    """
    parts = address.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        if not part.isdigit():
            return False
        if not 0 <= int(part) <= 255:
            return False
        if len(part) > 1 and part.startswith("0"):
            return False
    return True


def is_valid_hostname(hostname: str) -> bool:
    """Return True if *hostname* is a syntactically valid DNS hostname.

    Args:
        hostname: Hostname string, e.g. ``"api.example.com"``.

    Returns:
        True if the hostname conforms to basic DNS label rules; False otherwise.
    """
    import re

    if not hostname or len(hostname) > 253:
        return False
    hostname = hostname.rstrip(".")
    label_re = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?$")
    return all(label_re.match(label) for label in hostname.split("."))


def validate_non_negative(value: float, name: str = "value") -> float:
    """Assert *value* is non-negative (>= 0) and return it.

    Args:
        value: Numeric value to check.
        name: Human-readable name used in the error message.

    Returns:
        *value* unchanged.

    Raises:
        ValueError: If *value* is negative.
    """
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value
