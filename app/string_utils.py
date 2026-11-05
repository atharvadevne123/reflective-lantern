"""String manipulation helpers used throughout the Watt-Guard application."""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "camel_to_snake",
    "format_bytes",
    "format_number",
    "pad_left",
    "pad_right",
    "remove_whitespace",
    "sanitize_identifier",
    "slugify",
    "snake_to_camel",
    "title_case",
    "truncate",
]


def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """Return *text* truncated to *max_length* characters, appending *suffix*.

    Args:
        text: The input string to truncate.
        max_length: Maximum number of characters in the result (inclusive of suffix).
        suffix: Appended when truncation occurs. Defaults to ``"..."``.

    Returns:
        The original string if it fits; otherwise a truncated string ending with *suffix*.

    Raises:
        ValueError: If *max_length* is less than the length of *suffix*.
    """
    if max_length < len(suffix):
        raise ValueError(f"max_length ({max_length}) must be >= len(suffix) ({len(suffix)})")
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def slugify(text: str) -> str:
    """Convert *text* to a URL-safe lowercase slug.

    Non-ASCII characters are normalised and stripped.  Spaces and hyphens are
    collapsed into a single hyphen.

    Args:
        text: Arbitrary human-readable string.

    Returns:
        Lowercase slug suitable for URLs and identifiers.
    """
    normalised = unicodedata.normalize("NFKD", text)
    ascii_text = normalised.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_text.lower()
    slugged = re.sub(r"[^a-z0-9\s-]", "", lowered)
    return re.sub(r"[\s-]+", "-", slugged).strip("-")


def camel_to_snake(name: str) -> str:
    """Convert a CamelCase identifier to snake_case.

    Args:
        name: CamelCase string such as ``"MyClassName"`` or ``"HTTPResponseCode"``.

    Returns:
        snake_case version, e.g. ``"my_class_name"`` or ``"http_response_code"``.
    """
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def snake_to_camel(name: str) -> str:
    """Convert a snake_case identifier to lowerCamelCase.

    Args:
        name: snake_case string such as ``"my_variable_name"``.

    Returns:
        lowerCamelCase version, e.g. ``"myVariableName"``.
    """
    parts = name.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


def sanitize_identifier(name: str) -> str:
    """Return a safe Python-like identifier derived from *name*.

    Replaces non-alphanumeric characters with underscores and ensures the
    result does not start with a digit.

    Args:
        name: Arbitrary string to sanitise.

    Returns:
        Safe identifier string suitable for use as a variable or column name.
    """
    sanitised = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if sanitised and sanitised[0].isdigit():
        sanitised = "_" + sanitised
    return sanitised or "_"


def pad_left(text: str, width: int, fill: str = " ") -> str:
    """Left-pad *text* with *fill* characters to reach *width*.

    Args:
        text: String to pad.
        width: Minimum output width.
        fill: Single padding character. Defaults to a space.

    Returns:
        Original string when already >= *width*; padded string otherwise.

    Raises:
        ValueError: If *fill* is not exactly one character.
    """
    if len(fill) != 1:
        raise ValueError("fill must be exactly one character")
    return text.rjust(width, fill)


def pad_right(text: str, width: int, fill: str = " ") -> str:
    """Right-pad *text* with *fill* characters to reach *width*.

    Args:
        text: String to pad.
        width: Minimum output width.
        fill: Single padding character. Defaults to a space.

    Returns:
        Original string when already >= *width*; padded string otherwise.

    Raises:
        ValueError: If *fill* is not exactly one character.
    """
    if len(fill) != 1:
        raise ValueError("fill must be exactly one character")
    return text.ljust(width, fill)


def remove_whitespace(text: str) -> str:
    """Remove all whitespace characters from *text*.

    Args:
        text: Input string.

    Returns:
        String with all spaces, tabs, newlines etc. removed.
    """
    return re.sub(r"\s+", "", text)


def title_case(text: str) -> str:
    """Convert *text* to Title Case, preserving existing mixed case in words.

    Uses Python's built-in :meth:`str.title` for compatibility.

    Args:
        text: Arbitrary string.

    Returns:
        Title-cased version of *text*.
    """
    return text.title()


def format_bytes(size: int, precision: int = 2) -> str:
    """Return a human-readable string for a byte count.

    Args:
        size: Number of bytes (non-negative integer).
        precision: Decimal places for fractional values. Defaults to 2.

    Returns:
        String like ``"1.23 KB"`` or ``"4.56 MB"``.

    Raises:
        ValueError: If *size* is negative.
    """
    if size < 0:
        raise ValueError(f"size must be non-negative, got {size}")
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(size)
    for unit in units[:-1]:
        if value < 1024:
            return f"{value:.{precision}f} {unit}"
        value /= 1024
    return f"{value:.{precision}f} {units[-1]}"


def format_number(value: float, precision: int = 2, thousands_sep: str = ",") -> str:
    """Format a number with thousands separators and fixed decimal places.

    Args:
        value: Number to format.
        precision: Decimal places. Defaults to 2.
        thousands_sep: Separator character. Defaults to ``,``.

    Returns:
        Formatted string, e.g. ``"1,234.56"``.
    """
    formatted = f"{value:,.{precision}f}"
    if thousands_sep != ",":
        formatted = formatted.replace(",", thousands_sep)
    return formatted
