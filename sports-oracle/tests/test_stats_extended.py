"""Extended tests for sports-oracle app/stats.py MetricsCollector."""

from __future__ import annotations

import threading

import pytest

from app.stats import MetricsCollector, get_collector


class TestMetricsCollector:
    def test_initial_counters_zero(self) -> None:
        mc = MetricsCollector()
        snap = mc.snapshot()
        assert snap["predict_total"] == 0
        assert snap["predict_errors"] == 0

    def test_inc_increments_by_one(self) -> None:
        mc = MetricsCollector()
        mc.inc("predict_total")
        assert mc.snapshot()["predict_total"] == 1

    def test_inc_by_n(self) -> None:
        mc = MetricsCollector()
        mc.inc("health_checks", 5)
        assert mc.snapshot()["health_checks"] == 5

    def test_inc_new_counter(self) -> None:
        mc = MetricsCollector()
        mc.inc("custom_counter", 3)
        assert mc.snapshot()["custom_counter"] == 3

    def test_record_latency_increments_samples(self) -> None:
        mc = MetricsCollector()
        mc.record_latency(10.5)
        snap = mc.snapshot()
        assert snap["latency_samples"] == 1

    def test_latency_percentiles_populated(self) -> None:
        mc = MetricsCollector()
        for i in range(1, 101):
            mc.record_latency(float(i))
        snap = mc.snapshot()
        assert snap["latency_p50_ms"] > 0
        assert snap["latency_p95_ms"] >= snap["latency_p50_ms"]
        assert snap["latency_p99_ms"] >= snap["latency_p95_ms"]

    def test_no_latency_percentiles_zero(self) -> None:
        mc = MetricsCollector()
        snap = mc.snapshot()
        assert snap["latency_p50_ms"] == pytest.approx(0.0)
        assert snap["latency_p95_ms"] == pytest.approx(0.0)

    def test_uptime_seconds_non_negative(self) -> None:
        mc = MetricsCollector()
        snap = mc.snapshot()
        assert snap["uptime_seconds"] >= 0.0

    def test_snapshot_has_all_default_counters(self) -> None:
        mc = MetricsCollector()
        snap = mc.snapshot()
        for key in ("predict_total", "predict_errors", "health_checks", "drift_checks", "retrains"):
            assert key in snap

    def test_latency_capped_at_1000(self) -> None:
        mc = MetricsCollector()
        for i in range(1100):
            mc.record_latency(float(i))
        assert mc.snapshot()["latency_samples"] == 1000

    def test_thread_safety_concurrent_increments(self) -> None:
        mc = MetricsCollector()
        threads = [
            threading.Thread(target=lambda: mc.inc("predict_total"))
            for _ in range(100)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert mc.snapshot()["predict_total"] == 100


class TestGetCollector:
    def test_returns_metrics_collector(self) -> None:
        c = get_collector()
        assert isinstance(c, MetricsCollector)

    def test_same_instance_each_call(self) -> None:
        assert get_collector() is get_collector()
