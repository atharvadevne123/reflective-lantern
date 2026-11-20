"""Metrics collector tests."""
from __future__ import annotations

import pytest

import app.metrics_collector as mc


@pytest.fixture(autouse=True)
def _reset_metrics():
    mc.reset()
    yield
    mc.reset()


def test_increment_counter():
    mc.increment("predictions")
    mc.increment("predictions")
    snap = mc.snapshot()
    assert snap["counters"]["predictions"] == 2


def test_record_timing_p95():
    for i in range(100):
        mc.record_timing("predict_latency_ms", float(i))
    snap = mc.snapshot()
    timing = snap["timings"]["predict_latency_ms"]
    assert timing["p95_ms"] == pytest.approx(94.05, abs=1.0)
    assert timing["count"] == 100


def test_snapshot_empty():
    snap = mc.snapshot()
    assert snap["counters"] == {}
    assert snap["timings"] == {}


def test_reset_clears_state():
    mc.increment("x", 5)
    mc.record_timing("t", 10.0)
    mc.reset()
    snap = mc.snapshot()
    assert snap["counters"] == {}
    assert snap["timings"] == {}
