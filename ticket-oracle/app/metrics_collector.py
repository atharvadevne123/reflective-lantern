"""Telemetry counters and timing metrics for Ticket-Oracle.

Tracks request counts, prediction distributions, and latency histograms
in memory. Exposed via the /api/v1/metrics endpoint alongside model metrics.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Thread-safe in-process counter and latency tracker.

    Counters are monotonically increasing; histograms bucket latencies
    into 10 ms-wide slots up to 500 ms, then an overflow bucket.
    """

    _LATENCY_BUCKETS_MS = [10, 25, 50, 100, 200, 500]

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._latencies: list[float] = []

    def increment(self, name: str, by: int = 1) -> None:
        """Increment a named counter.

        Args:
            name: Counter name (e.g. 'predictions_total').
            by: Amount to add (default 1).
        """
        with self._lock:
            self._counters[name] += by

    def record_latency(self, elapsed_ms: float) -> None:
        """Record a request latency sample.

        Args:
            elapsed_ms: Elapsed time in milliseconds.
        """
        with self._lock:
            self._latencies.append(elapsed_ms)
            if len(self._latencies) > 10_000:
                self._latencies = self._latencies[-5_000:]

    def get_snapshot(self) -> dict[str, Any]:
        """Return a consistent snapshot of all metrics.

        Returns:
            Dictionary with counters and latency percentiles.
        """
        with self._lock:
            counters = dict(self._counters)
            lats = list(self._latencies)

        latency_stats: dict[str, float] = {}
        if lats:
            sorted_lats = sorted(lats)
            n = len(sorted_lats)
            latency_stats = {
                "p50_ms": sorted_lats[int(n * 0.50)],
                "p95_ms": sorted_lats[int(n * 0.95)],
                "p99_ms": sorted_lats[int(n * 0.99)],
                "max_ms": sorted_lats[-1],
                "count": n,
            }

        return {"counters": counters, "latency": latency_stats}


_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Return the singleton MetricsCollector instance."""
    return _collector
