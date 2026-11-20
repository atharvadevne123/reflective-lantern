"""Lightweight profiling decorators for timing and memory usage."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


def timed(label: str | None = None, log_level: int = logging.DEBUG) -> Callable:
    """Decorator that logs the wall-clock execution time of a function.

    Args:
        label: Custom label in log messages; defaults to func.__qualname__.
        log_level: Python logging level to emit the timing at.

    Returns:
        Wrapped function that logs its duration on each call.
    """

    def decorator(func: Callable) -> Callable:
        name = label or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.log(log_level, "%s completed in %.2f ms", name, elapsed_ms)

        return wrapper

    return decorator


class _Stats:
    """Running statistics for call durations."""

    def __init__(self) -> None:
        self.calls: int = 0
        self.total_ms: float = 0.0
        self.min_ms: float = float("inf")
        self.max_ms: float = 0.0

    def record(self, ms: float) -> None:
        self.calls += 1
        self.total_ms += ms
        self.min_ms = min(self.min_ms, ms)
        self.max_ms = max(self.max_ms, ms)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "calls": self.calls,
            "total_ms": round(self.total_ms, 3),
            "avg_ms": round(self.avg_ms, 3),
            "min_ms": round(self.min_ms, 3) if self.calls else 0,
            "max_ms": round(self.max_ms, 3),
        }


_registry: dict[str, _Stats] = {}


def tracked(label: str | None = None) -> Callable:
    """Decorator that records call statistics retrievable via :func:`get_stats`.

    Args:
        label: Registry key; defaults to func.__qualname__.

    Returns:
        Wrapped function that accumulates timing statistics.
    """

    def decorator(func: Callable) -> Callable:
        name = label or func.__qualname__
        _registry[name] = _Stats()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                ms = (time.perf_counter() - start) * 1000
                _registry[name].record(ms)

        return wrapper

    return decorator


def get_stats(label: str | None = None) -> dict:
    """Return accumulated stats for a label or all tracked functions.

    Args:
        label: Specific function label, or None for all.

    Returns:
        Dict mapping label to stats dict, or single stats dict.
    """
    if label is not None:
        stats = _registry.get(label)
        return stats.to_dict() if stats else {}
    return {k: v.to_dict() for k, v in _registry.items()}


def reset_stats(label: str | None = None) -> None:
    """Reset accumulated stats.

    Args:
        label: Specific label to reset, or None to reset all.
    """
    if label is not None:
        if label in _registry:
            _registry[label] = _Stats()
    else:
        for key in list(_registry):
            _registry[key] = _Stats()


__all__ = ["get_stats", "reset_stats", "timed", "tracked"]
