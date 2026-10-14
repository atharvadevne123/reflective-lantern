"""In-process metrics counters for monitoring and observability."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class ServiceMetrics:
    """Thread-safe counters for service observability.

    Attributes:
        start_time: Monotonic clock at service startup.
        prediction_count: Total predictions served.
        drift_alert_count: Total drift alerts triggered.
        error_count: Total unhandled errors.
        cache_hits: Prediction cache hits.
        cache_misses: Prediction cache misses.
    """

    start_time: float = field(default_factory=time.monotonic)
    prediction_count: int = 0
    drift_alert_count: int = 0
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def increment_predictions(self) -> None:
        """Atomically increment prediction counter."""
        with self._lock:
            self.prediction_count += 1

    def increment_drift_alerts(self, n: int = 1) -> None:
        """Atomically increment drift alert counter by n."""
        with self._lock:
            self.drift_alert_count += n

    def increment_errors(self) -> None:
        """Atomically increment error counter."""
        with self._lock:
            self.error_count += 1

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self.cache_misses += 1

    @property
    def uptime_seconds(self) -> float:
        """Return service uptime in seconds."""
        return round(time.monotonic() - self.start_time, 1)

    @property
    def cache_hit_rate(self) -> float:
        """Return cache hit rate as fraction 0-1."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def snapshot(self) -> dict:
        """Return a dict snapshot of all metrics.

        Returns:
            Dictionary of current metric values.
        """
        with self._lock:
            return {
                "uptime_seconds": self.uptime_seconds,
                "prediction_count": self.prediction_count,
                "drift_alert_count": self.drift_alert_count,
                "error_count": self.error_count,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": round(self.cache_hit_rate, 4),
            }


_service_metrics = ServiceMetrics()
