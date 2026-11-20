"""Shared utility functions for Social-Pulse."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")


def generate_request_id() -> str:
    """Return a new UUID4 string suitable for use as a request identifier.

    Returns:
        A hyphen-separated UUID4 string.
    """
    return str(uuid.uuid4())


@lru_cache(maxsize=1024)
def normalise_text(text: str) -> str:
    """Lowercase and collapse whitespace in *text*.

    Cached because the same post text may be processed multiple times
    during batch inference.

    Args:
        text: Raw post text.

    Returns:
        Lowercased string with collapsed whitespace.
    """
    return _WHITESPACE_RE.sub(" ", text.lower()).strip()


def content_hash(text: str) -> str:
    """Return a stable SHA-256 hex digest of the post text.

    Useful for deduplicating identical posts in the prediction log.

    Args:
        text: Raw post text.

    Returns:
        64-character hex string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clamp(value: float, lo: float, hi: float) -> float:
    """Return *value* clipped to the range [*lo*, *hi*].

    Args:
        value: The input float.
        lo: Lower bound (inclusive).
        hi: Upper bound (inclusive).

    Returns:
        The clamped float.
    """
    return max(lo, min(hi, value))


class Timer:
    """Context manager that measures elapsed wall-clock time in milliseconds.

    Usage::

        with Timer() as t:
            ...
        logger.debug("elapsed: %s ms", t.elapsed_ms)
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        logger.debug("Timer: %.2f ms", self.elapsed_ms)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide *numerator* by *denominator*, returning *default* on zero division.

    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value to return when *denominator* is zero.

    Returns:
        The quotient, or *default*.
    """
    if denominator == 0:
        return default
    return numerator / denominator


def truncate_text(text: str, max_chars: int = 280) -> str:
    """Truncate *text* to at most *max_chars* characters.

    If truncated, appends an ellipsis so callers know the text was cut.

    Args:
        text: Input string.
        max_chars: Maximum allowed length including the ellipsis.

    Returns:
        The original string if short enough, otherwise a truncated version.
    """
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def batch_normalise(texts: list[str]) -> list[str]:
    """Apply :func:`normalise_text` to every string in *texts*.

    Args:
        texts: List of raw post strings.

    Returns:
        New list of normalised strings in the same order.
    """
    return [normalise_text(t) for t in texts]


def sanitize_text(text: str, max_length: int = 2000) -> str:
    """Remove leading/trailing whitespace and enforce a maximum length.

    Args:
        text: Input text.
        max_length: Maximum character count after stripping.

    Returns:
        Stripped and optionally truncated text.
    """
    stripped = text.strip()
    return stripped[:max_length]


def safe_json_dumps(data: Any, default: str = "{}") -> str:
    """Serialise *data* to a JSON string, returning *default* on failure.

    Args:
        data: Python object to serialise.
        default: Fallback string if serialisation fails.

    Returns:
        JSON string, or *default* if an error occurs.
    """
    try:
        return json.dumps(data)
    except (TypeError, ValueError):
        return default
