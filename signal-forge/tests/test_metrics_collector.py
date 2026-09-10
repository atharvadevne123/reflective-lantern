"""Tests for the metrics collector module."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.metrics_collector import Counter, Histogram, MetricsRegistry, get_registry


def test_counter_increments():
    c = Counter(name="test_counter")
    c.inc()
    c.inc(5.0)
    assert c.value == 6.0


def test_histogram_records_mean():
    h = Histogram(name="test_hist")
    h.observe(0.1)
    h.observe(0.3)
    assert h.count == 2
    assert h.mean == pytest.approx(0.2)


def test_histogram_empty_mean():
    h = Histogram(name="empty")
    assert h.mean == 0.0


def test_registry_counter_singleton():
    reg = MetricsRegistry()
    c1 = reg.counter("req")
    c2 = reg.counter("req")
    assert c1 is c2


def test_registry_snapshot_structure():
    reg = MetricsRegistry()
    reg.counter("requests").inc(3)
    reg.histogram("latency").observe(0.05)
    snap = reg.snapshot()
    assert "counters" in snap
    assert "histograms" in snap
    assert snap["counters"]["requests"] == 3.0


@pytest.mark.parametrize("amount", [1.0, 10.0, 0.5])
def test_counter_multiple_amounts(amount):
    c = Counter(name="multi")
    c.inc(amount)
    assert c.value == amount


def test_get_registry_returns_same_instance():
    r1 = get_registry()
    r2 = get_registry()
    assert r1 is r2
