"""Extended tests for ticket-oracle app/metrics_collector.py."""

from __future__ import annotations

import threading

from app.metrics_collector import MetricsCollector


class TestMetricsCollectorCounters:
    def test_increment_basic(self) -> None:
        mc = MetricsCollector()
        mc.increment("hits")
        snap = mc.get_snapshot()
        assert snap["counters"]["hits"] == 1

    def test_increment_by_n(self) -> None:
        mc = MetricsCollector()
        mc.increment("requests", by=5)
        snap = mc.get_snapshot()
        assert snap["counters"]["requests"] == 5

    def test_multiple_counters(self) -> None:
        mc = MetricsCollector()
        mc.increment("a")
        mc.increment("b")
        snap = mc.get_snapshot()
        assert snap["counters"]["a"] == 1
        assert snap["counters"]["b"] == 1

    def test_counter_accumulates(self) -> None:
        mc = MetricsCollector()
        for _ in range(10):
            mc.increment("tick")
        snap = mc.get_snapshot()
        assert snap["counters"]["tick"] == 10

    def test_thread_safety(self) -> None:
        mc = MetricsCollector()
        threads = [threading.Thread(target=mc.increment, args=("c",)) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        snap = mc.get_snapshot()
        assert snap["counters"]["c"] == 100


class TestMetricsCollectorLatency:
    def test_record_latency_stored(self) -> None:
        mc = MetricsCollector()
        mc.record_latency(50.0)
        snap = mc.get_snapshot()
        assert "p50_ms" in snap.get("latency", snap)

    def test_snapshot_returns_dict(self) -> None:
        mc = MetricsCollector()
        assert isinstance(mc.get_snapshot(), dict)

    def test_empty_snapshot_has_counters(self) -> None:
        mc = MetricsCollector()
        snap = mc.get_snapshot()
        assert "counters" in snap

    def test_latency_percentiles_computed(self) -> None:
        mc = MetricsCollector()
        for i in range(1, 101):
            mc.record_latency(float(i))
        snap = mc.get_snapshot()
        latency = snap.get("latency", snap)
        assert "p50_ms" in latency
        assert "p95_ms" in latency

    def test_p95_greater_than_p50(self) -> None:
        mc = MetricsCollector()
        for i in range(1, 101):
            mc.record_latency(float(i))
        snap = mc.get_snapshot()
        latency = snap.get("latency", snap)
        assert latency["p95_ms"] >= latency["p50_ms"]
