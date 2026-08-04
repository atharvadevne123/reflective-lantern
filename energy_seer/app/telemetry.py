"""Request and prediction telemetry counters."""

from __future__ import annotations

from collections import defaultdict

_counters: dict[str, int] = defaultdict(int)


def increment(name: str, amount: int = 1) -> None:
    _counters[name] += amount


def get(name: str) -> int:
    return _counters[name]


def snapshot() -> dict[str, int]:
    return dict(_counters)


def reset(name: str | None = None) -> None:
    if name:
        _counters[name] = 0
    else:
        _counters.clear()


def increment_by(name: str, amount: int) -> None:
    """Increment counter *name* by *amount*.

    Args:
        name: Counter name.
        amount: Amount to add (must be non-negative).

    Raises:
        ValueError: If *amount* is negative.
    """
    if amount < 0:
        raise ValueError(f"amount must be non-negative, got {amount}")
    _counters[name] += amount


def all_above(threshold: int) -> dict[str, int]:
    """Return all counters whose value exceeds *threshold*.

    Args:
        threshold: Minimum value to include in result.

    Returns:
        Dict of counter name → value for counters above the threshold.
    """
    return {k: v for k, v in _counters.items() if v > threshold}


def top_n(n: int = 5) -> list[tuple[str, int]]:
    """Return the *n* counters with the highest values.

    Args:
        n: Number of top entries to return (default 5).

    Returns:
        List of (name, value) tuples sorted descending by value.
    """
    return sorted(_counters.items(), key=lambda kv: kv[1], reverse=True)[:n]


__all__ = [
    "all_above",
    "get",
    "increment",
    "increment_by",
    "reset",
    "snapshot",
    "top_n",
]
