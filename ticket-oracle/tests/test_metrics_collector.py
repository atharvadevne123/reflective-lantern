"""Tests for ticket-oracle app/metrics_collector.py."""

from __future__ import annotations

import time

import pytest

from app.metrics_collector import MetricsCollector


class TestMetricsCollector:
    def test_init(self) -> None:
        mc = MetricsCollector()
        assert mc is not None

    def test_increment_counter(self) -> None:
        mc = MetricsCollector()
        mc.increment("predictions")
        mc.increment("predictions")
        assert mc._counters["predictions"] == 2

    def test_increment_starts_at_zero(self) -> None:
        mc = MetricsCollector()
        mc.increment("new_counter")
        assert mc._counters["new_counter"] == 1

    def test_record_latency(self) -> None:
        mc = MetricsCollector()
        mc.record_latency(50.0)
        assert len(mc._latencies) == 1

    def test_get_summary_returns_dict(self) -> None:
        mc = MetricsCollector()
        mc.increment("predictions")
        mc.record_latency(25.0)
        summary = mc.get_summary()
        assert isinstance(summary, dict)

    def test_thread_safe_increments(self) -> None:
        import threading

        mc = MetricsCollector()
        threads = [threading.Thread(target=mc.increment, args=("counter",)) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert mc._counters["counter"] == 100

    def test_multiple_counters(self) -> None:
        mc = MetricsCollector()
        mc.increment("a")
        mc.increment("b")
        mc.increment("a")
        assert mc._counters["a"] == 2
        assert mc._counters["b"] == 1

    @pytest.mark.parametrize("latency_ms", [5.0, 25.0, 100.0, 250.0, 600.0])
    def test_various_latencies(self, latency_ms: float) -> None:
        mc = MetricsCollector()
        mc.record_latency(latency_ms)
        assert len(mc._latencies) == 1
