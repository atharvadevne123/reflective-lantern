"""Extended tests for Sports-Oracle MetricsCollector."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.stats import MetricsCollector


def test_metrics_collector_initial_counters():
    mc = MetricsCollector()
    snap = mc.snapshot()
    assert snap["predict_total"] == 0
    assert snap["predict_errors"] == 0
    assert snap["health_checks"] == 0


def test_inc_increments_counter():
    mc = MetricsCollector()
    mc.inc("predict_total")
    snap = mc.snapshot()
    assert snap["predict_total"] == 1


def test_inc_multiple_times():
    mc = MetricsCollector()
    mc.inc("drift_checks", 3)
    snap = mc.snapshot()
    assert snap["drift_checks"] == 3


def test_record_latency_increases_samples():
    mc = MetricsCollector()
    mc.record_latency(10.0)
    mc.record_latency(20.0)
    snap = mc.snapshot()
    assert snap["latency_samples"] == 2


def test_snapshot_has_uptime():
    mc = MetricsCollector()
    snap = mc.snapshot()
    assert snap["uptime_seconds"] >= 0


def test_snapshot_latency_p50_zero_when_no_samples():
    mc = MetricsCollector()
    snap = mc.snapshot()
    assert snap["latency_p50_ms"] == 0.0


def test_snapshot_latency_p50_correct():
    mc = MetricsCollector()
    for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
        mc.record_latency(v)
    snap = mc.snapshot()
    assert snap["latency_p50_ms"] == pytest.approx(30.0)


def test_snapshot_latency_p99_less_or_equal_max():
    mc = MetricsCollector()
    values = [float(i) for i in range(100)]
    for v in values:
        mc.record_latency(v)
    snap = mc.snapshot()
    assert snap["latency_p99_ms"] <= max(values)


def test_inc_custom_counter():
    mc = MetricsCollector()
    mc.inc("custom_counter")
    snap = mc.snapshot()
    assert snap["custom_counter"] == 1


def test_latency_buffer_bounded():
    mc = MetricsCollector()
    for i in range(1200):
        mc.record_latency(float(i))
    snap = mc.snapshot()
    assert snap["latency_samples"] <= 1000


def test_thread_safety():
    mc = MetricsCollector()
    errors = []

    def worker():
        try:
            for _ in range(100):
                mc.inc("predict_total")
                mc.record_latency(1.0)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    snap = mc.snapshot()
    assert snap["predict_total"] == 500


@pytest.mark.parametrize("counter", ["predict_total", "predict_errors", "health_checks"])
def test_counters_start_at_zero(counter):
    mc = MetricsCollector()
    snap = mc.snapshot()
    assert snap[counter] == 0
