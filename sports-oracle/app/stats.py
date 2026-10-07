"""In-process request statistics collector."""

from __future__ import annotations

import threading
import time
from typing import Any


class MetricsCollector:
    """Thread-safe request and prediction metrics counters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {
            "predict_total": 0,
            "predict_errors": 0,
            "health_checks": 0,
            "drift_checks": 0,
            "retrains": 0,
        }
        self._latencies: list[float] = []
        self._start = time.monotonic()

    def inc(self, counter: str, n: int = 1) -> None:
        """Increment a named counter."""
        with self._lock:
            self._counters[counter] = self._counters.get(counter, 0) + n

    def record_latency(self, latency_ms: float) -> None:
        """Record a prediction latency sample."""
        with self._lock:
            self._latencies.append(latency_ms)
            if len(self._latencies) > 1000:
                self._latencies.pop(0)

    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot of current metrics."""
        with self._lock:
            latencies = list(self._latencies)
            counters = dict(self._counters)
        uptime_s = round(time.monotonic() - self._start, 1)
        p50 = p95 = p99 = 0.0
        if latencies:
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            p50 = sorted_lat[int(n * 0.50)]
            p95 = sorted_lat[min(int(n * 0.95), n - 1)]
            p99 = sorted_lat[min(int(n * 0.99), n - 1)]
        return {
            **counters,
            "uptime_seconds": uptime_s,
            "latency_p50_ms": round(p50, 2),
            "latency_p95_ms": round(p95, 2),
            "latency_p99_ms": round(p99, 2),
            "latency_samples": len(latencies),
        }


_collector = MetricsCollector()


def get_collector() -> MetricsCollector:
    return _collector
