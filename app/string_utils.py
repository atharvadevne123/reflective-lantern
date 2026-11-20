"""String manipulation helpers used throughout the Watt-Guard application."""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "camel_to_snake",
    "sanitize_identifier",
    "slugify",
    "snake_to_camel",
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
