"""MetricsCollector tests for Sports-Oracle."""

from __future__ import annotations

import threading


def test_increment_counter():
    from app.stats import MetricsCollector

    c = MetricsCollector()
    c.inc("predict_total")
    c.inc("predict_total")
    snap = c.snapshot()
    assert snap["predict_total"] == 2


def test_record_latency_percentiles():
    from app.stats import MetricsCollector

    c = MetricsCollector()
    for i in range(100):
        c.record_latency(float(i))
    snap = c.snapshot()
    assert snap["latency_p50_ms"] > 0
    assert snap["latency_p95_ms"] >= snap["latency_p50_ms"]
    assert snap["latency_p99_ms"] >= snap["latency_p95_ms"]


def test_snapshot_returns_uptime():
    from app.stats import MetricsCollector

    c = MetricsCollector()
    snap = c.snapshot()
    assert "uptime_seconds" in snap
    assert snap["uptime_seconds"] >= 0


def test_thread_safety():
    from app.stats import MetricsCollector

    c = MetricsCollector()
    threads = [threading.Thread(target=lambda: c.inc("predict_total")) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = c.snapshot()
    assert snap["predict_total"] == 50


def test_latency_buffer_capped_at_1000():
    from app.stats import MetricsCollector

    c = MetricsCollector()
    for i in range(1200):
        c.record_latency(float(i))
    snap = c.snapshot()
    assert snap["latency_samples"] <= 1000
