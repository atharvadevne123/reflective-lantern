"""Extended metrics collector tests with parametrize and thread-safety checks."""
from __future__ import annotations

import threading
import time

import pytest

from app.stats import MetricsCollector


def test_initial_counters_zero() -> None:
    mc = MetricsCollector()
    snap = mc.snapshot()
    assert snap["predict_total"] == 0
    assert snap["predict_errors"] == 0
    assert snap["health_checks"] == 0


@pytest.mark.parametrize("counter,n", [
    ("predict_total", 1),
    ("predict_errors", 5),
    ("health_checks", 10),
    ("drift_checks", 3),
    ("retrains", 2),
])
def test_inc_increments_counter(counter: str, n: int) -> None:
    mc = MetricsCollector()
    mc.inc(counter, n)
    snap = mc.snapshot()
    assert snap[counter] == n


def test_inc_unknown_counter_creates_it() -> None:
    mc = MetricsCollector()
    mc.inc("custom_counter", 7)
    snap = mc.snapshot()
    assert snap["custom_counter"] == 7


def test_inc_multiple_times_accumulates() -> None:
    mc = MetricsCollector()
    mc.inc("predict_total")
    mc.inc("predict_total")
    mc.inc("predict_total", 3)
    snap = mc.snapshot()
    assert snap["predict_total"] == 5


def test_record_latency_stores_samples() -> None:
    mc = MetricsCollector()
    mc.record_latency(10.0)
    mc.record_latency(20.0)
    snap = mc.snapshot()
    assert snap["latency_samples"] == 2


def test_record_latency_p50_bounded() -> None:
    mc = MetricsCollector()
    for i in range(100):
        mc.record_latency(float(i))
    snap = mc.snapshot()
    assert 40.0 <= snap["latency_p50_ms"] <= 60.0


def test_record_latency_p95_at_least_p50() -> None:
    mc = MetricsCollector()
    for i in range(100):
        mc.record_latency(float(i))
    snap = mc.snapshot()
    assert snap["latency_p95_ms"] >= snap["latency_p50_ms"]


def test_record_latency_p99_at_least_p95() -> None:
    mc = MetricsCollector()
    for i in range(100):
        mc.record_latency(float(i))
    snap = mc.snapshot()
    assert snap["latency_p99_ms"] >= snap["latency_p95_ms"]


def test_snapshot_no_latency_all_zero() -> None:
    mc = MetricsCollector()
    snap = mc.snapshot()
    assert snap["latency_p50_ms"] == 0.0
    assert snap["latency_p95_ms"] == 0.0
    assert snap["latency_p99_ms"] == 0.0
    assert snap["latency_samples"] == 0


def test_latency_buffer_capped_at_1000() -> None:
    mc = MetricsCollector()
    for i in range(1200):
        mc.record_latency(float(i))
    snap = mc.snapshot()
    assert snap["latency_samples"] == 1000


def test_uptime_seconds_increases() -> None:
    mc = MetricsCollector()
    snap1 = mc.snapshot()
    time.sleep(0.05)
    snap2 = mc.snapshot()
    assert snap2["uptime_seconds"] >= snap1["uptime_seconds"]


def test_snapshot_has_all_expected_keys() -> None:
    mc = MetricsCollector()
    snap = mc.snapshot()
    for key in ("predict_total", "predict_errors", "health_checks", "drift_checks",
                "retrains", "uptime_seconds", "latency_p50_ms", "latency_p95_ms",
                "latency_p99_ms", "latency_samples"):
        assert key in snap


def test_thread_safe_concurrent_inc() -> None:
    mc = MetricsCollector()
    n_threads = 50
    n_per_thread = 100
    threads = [
        threading.Thread(target=lambda: [mc.inc("predict_total") for _ in range(n_per_thread)])
        for _ in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = mc.snapshot()
    assert snap["predict_total"] == n_threads * n_per_thread


@pytest.mark.parametrize("latency", [0.0, 1.0, 100.0, 9999.99])
def test_record_single_latency(latency: float) -> None:
    mc = MetricsCollector()
    mc.record_latency(latency)
    snap = mc.snapshot()
    assert snap["latency_samples"] == 1
    assert snap["latency_p50_ms"] == pytest.approx(latency, rel=1e-5)
