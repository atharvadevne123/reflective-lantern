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
