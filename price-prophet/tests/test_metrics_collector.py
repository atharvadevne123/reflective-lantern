"""Tests for app/utils/metrics_collector.py."""

from __future__ import annotations


def test_increment_default():
    from app.utils.metrics_collector import MetricsCollector

    mc = MetricsCollector()
    mc.increment("hits")
    assert mc.counter("hits") == 1


def test_increment_by():
    from app.utils.metrics_collector import MetricsCollector

    mc = MetricsCollector()
    mc.increment("hits", by=5)
    assert mc.counter("hits") == 5


def test_counter_zero_for_unseen():
    from app.utils.metrics_collector import MetricsCollector

    mc = MetricsCollector()
    assert mc.counter("nope") == 0


def test_record_timing_avg():
    from app.utils.metrics_collector import MetricsCollector

    mc = MetricsCollector()
    mc.record_timing("req", 1.0)
    mc.record_timing("req", 3.0)
    assert abs(mc.avg_timing("req") - 2.0) < 1e-9


def test_avg_timing_empty():
    from app.utils.metrics_collector import MetricsCollector

    mc = MetricsCollector()
    assert mc.avg_timing("missing") == 0.0


def test_snapshot_structure():
    from app.utils.metrics_collector import MetricsCollector

    mc = MetricsCollector()
    mc.increment("x")
    snap = mc.snapshot()
    assert "counters" in snap
    assert "avg_timings" in snap


def test_reset_clears():
    from app.utils.metrics_collector import MetricsCollector

    mc = MetricsCollector()
    mc.increment("x", 10)
    mc.reset()
    assert mc.counter("x") == 0


def test_metrics_collector_timing_average() -> None:
    from app.utils.metrics_collector import MetricsCollector

    mc = MetricsCollector()
    mc.record_timing("op", 10.0)
    mc.record_timing("op", 20.0)
    snap = mc.snapshot()
    avg = snap["avg_timings"].get("op", 0.0)
    assert abs(avg - 15.0) < 1e-6


def test_metrics_collector_counter_increment() -> None:
    from app.utils.metrics_collector import MetricsCollector

    mc = MetricsCollector()
    mc.increment("requests", 5)
    mc.increment("requests", 3)
    assert mc.counter("requests") == 8


def test_metrics_collector_counter_nonexistent_returns_zero() -> None:
    from app.utils.metrics_collector import MetricsCollector

    mc = MetricsCollector()
    assert mc.counter("nonexistent") == 0


def test_metrics_collector_multiple_timings() -> None:
    from app.utils.metrics_collector import MetricsCollector

    mc = MetricsCollector()
    for t in [1.0, 2.0, 3.0, 4.0]:
        mc.record_timing("task", t)
    assert abs(mc.avg_timing("task") - 2.5) < 1e-9


def test_metrics_collector_reset_clears_timings() -> None:
    from app.utils.metrics_collector import MetricsCollector

    mc = MetricsCollector()
    mc.record_timing("op", 5.0)
    mc.reset()
    assert mc.avg_timing("op") == 0.0


def test_metrics_collector_snapshot_has_both_sections() -> None:
    from app.utils.metrics_collector import MetricsCollector

    mc = MetricsCollector()
    mc.increment("calls")
    mc.record_timing("latency", 0.1)
    snap = mc.snapshot()
    assert "counters" in snap
    assert "avg_timings" in snap
    assert snap["counters"]["calls"] == 1


def test_metrics_collector_increment_zero_by() -> None:
    from app.utils.metrics_collector import MetricsCollector

    mc = MetricsCollector()
    mc.increment("evt", by=0)
    assert mc.counter("evt") == 0
