"""Prometheus-compatible metrics collector for Signal-Forge."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Counter:
    """Simple monotonic counter."""

    name: str
    _value: float = field(default=0.0, init=False, repr=False)

    def inc(self, amount: float = 1.0) -> None:
        """Increment the counter by `amount`."""
        self._value += amount

    @property
    def value(self) -> float:
        return self._value


@dataclass
class Histogram:
    """Simple latency histogram tracking count and sum."""

    name: str
    _count: int = field(default=0, init=False, repr=False)
    _sum: float = field(default=0.0, init=False, repr=False)

    def observe(self, value: float) -> None:
        """Record one observation."""
        self._count += 1
        self._sum += value

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float:
        return self._sum / self._count if self._count else 0.0


class MetricsRegistry:
    """Central registry for all Signal-Forge metrics."""

    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}

    def counter(self, name: str) -> Counter:
        """Get or create a named counter."""
        if name not in self._counters:
            self._counters[name] = Counter(name=name)
        return self._counters[name]

    def histogram(self, name: str) -> Histogram:
        """Get or create a named histogram."""
        if name not in self._histograms:
            self._histograms[name] = Histogram(name=name)
        return self._histograms[name]

    def snapshot(self) -> dict[str, Any]:
        """Return a point-in-time snapshot of all metric values."""
        return {
            "counters": {n: c.value for n, c in self._counters.items()},
            "histograms": {
                n: {"count": h.count, "mean_ms": round(h.mean * 1000, 2)}
                for n, h in self._histograms.items()
            },
        }


_registry = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    """Return the global metrics registry."""
    return _registry
