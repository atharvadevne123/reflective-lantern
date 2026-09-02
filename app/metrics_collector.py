"""In-process metrics collection: counters, gauges, and histograms.

Designed as a lightweight alternative to Prometheus client for services that
export metrics via a custom endpoint or periodic flush.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "get_registry",
]


class Counter:
    """Monotonically increasing counter."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._value: float = 0.0
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0) -> None:
        """Increment the counter by *amount*; raises ValueError for negative amounts."""
        if amount < 0:
            raise ValueError("Counter can only increase")
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        """Return the current counter value."""
        with self._lock:
            return self._value

    def reset(self) -> None:
        """Reset the counter to zero."""
        with self._lock:
            self._value = 0.0


class Gauge:
    """Gauge that can go up or down."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._value: float = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        """Set the gauge to an absolute *value*."""
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0) -> None:
        """Increment the gauge by *amount*."""
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        """Decrement the gauge by *amount*."""
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        """Return the current gauge value."""
        with self._lock:
            return self._value


@dataclass
class Histogram:
    """Fixed-bucket histogram for latency or size distributions."""

    name: str
    description: str = ""
    buckets: list[float] = field(default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0])

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: list[int] = [0] * len(self.buckets)
        self._sum: float = 0.0
        self._total: int = 0

    def observe(self, value: float) -> None:
        """Record *value* in the appropriate histogram buckets."""
        with self._lock:
            self._sum += value
            self._total += 1
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    self._counts[i] += 1

    @property
    def sum(self) -> float:
        """Return the sum of all observed values."""
        with self._lock:
            return self._sum

    @property
    def count(self) -> int:
        """Return the total number of observations."""
        with self._lock:
            return self._total

    def percentile(self, p: float) -> float | None:
        """Estimate the *p*-th percentile (0-1) from bucket boundaries."""
        with self._lock:
            if self._total == 0:
                return None
            target = math.ceil(p * self._total)
            cumulative = 0
            for bound, cnt in zip(self.buckets, self._counts, strict=False):
                cumulative += cnt
                if cumulative >= target:
                    return bound
            return self.buckets[-1]


class MetricsRegistry:
    """Central registry for named metrics."""

    def __init__(self) -> None:
        self._metrics: dict[str, object] = {}

    def counter(self, name: str, description: str = "") -> Counter:
        """Return an existing counter by *name*, creating it on first access."""
        if name not in self._metrics:
            self._metrics[name] = Counter(name, description)
        return self._metrics[name]  # type: ignore[return-value]

    def gauge(self, name: str, description: str = "") -> Gauge:
        """Return an existing gauge by *name*, creating it on first access."""
        if name not in self._metrics:
            self._metrics[name] = Gauge(name, description)
        return self._metrics[name]  # type: ignore[return-value]

    def histogram(self, name: str, description: str = "", buckets: list[float] | None = None) -> Histogram:
        """Return an existing histogram by *name*, creating it on first access."""
        if name not in self._metrics:
            kwargs = {"name": name, "description": description}
            if buckets is not None:
                kwargs["buckets"] = buckets
            self._metrics[name] = Histogram(**kwargs)
        return self._metrics[name]  # type: ignore[return-value]

    def all_metrics(self) -> dict[str, object]:
        return dict(self._metrics)


_default = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    """Return the default global metrics registry."""
    return _default
