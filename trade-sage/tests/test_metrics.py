"""Tests for service metrics counters."""
from __future__ import annotations

import threading
import time

import pytest

from app.metrics import ServiceMetrics


def test_increment_predictions():
    m = ServiceMetrics()
    m.increment_predictions()
    m.increment_predictions()
    assert m.prediction_count == 2


def test_increment_drift_alerts():
    m = ServiceMetrics()
    m.increment_drift_alerts(3)
    assert m.drift_alert_count == 3


def test_cache_hit_rate_zero_when_empty():
    m = ServiceMetrics()
    assert m.cache_hit_rate == 0.0


def test_cache_hit_rate_calculation():
    m = ServiceMetrics()
    m.record_cache_hit()
    m.record_cache_hit()
    m.record_cache_miss()
    assert abs(m.cache_hit_rate - 2 / 3) < 0.01


def test_uptime_positive():
    m = ServiceMetrics()
    assert m.uptime_seconds >= 0.0


def test_snapshot_keys():
    m = ServiceMetrics()
    snap = m.snapshot()
    assert "prediction_count" in snap
    assert "drift_alert_count" in snap
    assert "cache_hit_rate" in snap
    assert "uptime_seconds" in snap


def test_thread_safety():
    m = ServiceMetrics()

    def worker():
        for _ in range(100):
            m.increment_predictions()

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert m.prediction_count == 1000
