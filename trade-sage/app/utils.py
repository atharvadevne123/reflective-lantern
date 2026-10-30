"""General-purpose utility functions for Trade-Sage."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


def safe_round(value: float | None, ndigits: int = 4) -> float | None:
    """Round a float to ndigits, returning None if the input is None.

    Args:
        value: Numeric value to round, or None.
        ndigits: Decimal places to round to.

    Returns:
        Rounded float or None.
    """
    if value is None:
        return None
    return round(value, ndigits)


def flatten_dict(d: dict[str, Any], prefix: str = "", sep: str = ".") -> dict[str, Any]:
    """Recursively flatten a nested dict into dot-separated key paths.

    Args:
        d: Input dict, possibly nested.
        prefix: Leading key fragment for recursion (empty on first call).
        sep: Separator between key levels.

    Returns:
        Flat dict with concatenated keys.

    Example:
        >>> flatten_dict({"a": {"b": 1, "c": 2}, "d": 3})
        {"a.b": 1, "a.c": 2, "d": 3}
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        full_key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, prefix=full_key, sep=sep))
        else:
            out[full_key] = v
    return out


def hash_payload(data: str) -> str:
    """Return a short SHA-256 hex digest for use as a cache key.

    Args:
        data: String payload to hash.

    Returns:
        First 16 hex characters of the SHA-256 digest.
    """
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value to the inclusive range [lo, hi].

    Args:
        value: Input value.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Value clamped to [lo, hi].
    """
    return max(lo, min(hi, value))


def truncate_string(text: str, max_len: int = 80, suffix: str = "...") -> str:
    """Truncate a string to max_len characters, appending suffix if cut.

    Args:
        text: Input string.
        max_len: Maximum length of the returned string including the suffix.
        suffix: String appended when truncation occurs.

    Returns:
        Original string if short enough, otherwise truncated with suffix.
    """
    if len(text) <= max_len:
        return text
    cut = max(0, max_len - len(suffix))
    return text[:cut] + suffix


def retry_with_backoff(
    fn: object,
    retries: int = 3,
    base_delay: float = 0.5,
    exceptions: tuple = (Exception,),
) -> object:
    """Call *fn* up to *retries* times with exponential backoff on failure.

    Args:
        fn: Zero-argument callable to invoke.
        retries: Maximum number of attempts (including the first).
        base_delay: Initial delay in seconds; doubles on each retry.
        exceptions: Exception types that trigger a retry.

    Returns:
        Return value of *fn* on first success.

    Raises:
        The last exception when all attempts are exhausted.
    """
    import time

    last_exc: Exception | None = None
    delay = base_delay
    for attempt in range(retries):
        try:
            return fn()
        except exceptions as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
    raise last_exc  # type: ignore[misc]


def percentage_change(old: float, new: float) -> float | None:
    """Compute (new - old) / abs(old) * 100 safely.

    Args:
        old: Reference value.
        new: New value.

    Returns:
        Percentage change, or None when old is zero.
    """
    if abs(old) < 1e-12:
        return None
    return round((new - old) / abs(old) * 100, 4)
