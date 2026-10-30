"""General-purpose utility helpers for Property-Sage.

Provides formatting, clamping, and text manipulation helpers used
across the application.
"""

from __future__ import annotations

import math
import re


def format_currency(amount: float, symbol: str = "$", decimals: int = 2) -> str:
    """Format a numeric amount as a currency string.

    Args:
        amount: Numeric value to format.
        symbol: Currency symbol prefix.
        decimals: Decimal places to include.

    Returns:
        Formatted string such as "$1,234.56".
    """
    rounded = round(amount, decimals)
    int_part, *frac = f"{abs(rounded):.{decimals}f}".split(".")
    groups = []
    while len(int_part) > 3:
        groups.insert(0, int_part[-3:])
        int_part = int_part[:-3]
    groups.insert(0, int_part)
    formatted = ",".join(groups)
    if decimals > 0:
        formatted += "." + frac[0]
    sign = "-" if rounded < 0 else ""
    return f"{sign}{symbol}{formatted}"


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi].

    Args:
        value: Input value.
        lo: Lower bound (inclusive).
        hi: Upper bound (inclusive).

    Returns:
        Value clamped to the given range.
    """
    return max(lo, min(hi, value))


def slugify(text: str) -> str:
    """Convert a string to a URL-safe slug.

    Args:
        text: Input string.

    Returns:
        Lowercased slug with hyphens replacing non-alphanumeric characters.
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s_]+", "-", text)


def percentage_change(old: float, new: float) -> float:
    """Compute percentage change from old to new.

    Args:
        old: Original value.
        new: New value.

    Returns:
        Percentage change as a float (e.g. 10.0 for a 10% increase).
        Returns 0.0 if old is zero.
    """
    if old == 0:
        return 0.0
    return round((new - old) / abs(old) * 100, 4)


def is_finite_positive(value: float) -> bool:
    """Return True if value is a finite, positive number.

    Args:
        value: Number to check.

    Returns:
        True if value > 0 and is finite.
    """
    return math.isfinite(value) and value > 0


def hash_payload(data: dict, length: int = 8) -> str:
    """Compute a short deterministic hash of a JSON-serialisable dict.

    Args:
        data: Dict to hash.
        length: Number of hex characters to return.

    Returns:
        Lowercase hex string of the specified length.
    """
    import hashlib
    import json as _json
    serialised = _json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialised.encode()).hexdigest()[:length]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide numerator by denominator, returning default on zero denominator.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Value to return when denominator is zero.

    Returns:
        Division result or default.
    """
    if denominator == 0:
        return default
    return numerator / denominator
