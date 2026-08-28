"""Lightweight profiling decorators for timing and memory usage."""

from __future__ import annotations

import functools
import logging
import time
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def timed(label: Optional[str] = None, log_level: int = logging.DEBUG) -> Callable:
    """Decorator that logs the wall-clock execution time of a function.

    Args:
        label: Custom label in log messages; defaults to func.__qualname__.
        log_level: Python logging level to emit the timing at.

    Returns:
        Wrapped function that logs its duration on each call.
    """
    def decorator(func: Callable) -> Callable:
        """Wrap *func* to log its duration on every call."""
        name = label or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # type: ignore[return]
            """Execute *func* and log elapsed time."""
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
        """Initialise counters for call count, total time, min, and max."""
        self.calls: int = 0
        self.total_ms: float = 0.0
        self.min_ms: float = float("inf")
        self.max_ms: float = 0.0

    def record(self, ms: float) -> None:
        """Accumulate a single timing observation of *ms* milliseconds."""
        self.calls += 1
        self.total_ms += ms
        self.min_ms = min(self.min_ms, ms)
        self.max_ms = max(self.max_ms, ms)

    @property
    def avg_ms(self) -> float:
        """Return mean duration in milliseconds; 0.0 if no calls recorded."""
        return self.total_ms / self.calls if self.calls else 0.0

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "calls": self.calls,
            "total_ms": round(self.total_ms, 3),
            "avg_ms": round(self.avg_ms, 3),
            "min_ms": round(self.min_ms, 3) if self.calls else 0,
            "max_ms": round(self.max_ms, 3),
        }


_registry: Dict[str, _Stats] = {}


def tracked(label: Optional[str] = None) -> Callable:
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


def get_stats(label: Optional[str] = None) -> Dict:
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


def reset_stats(label: Optional[str] = None) -> None:
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
