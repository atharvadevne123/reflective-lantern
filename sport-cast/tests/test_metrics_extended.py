"""Extended tests for sport-cast app/metrics_collector.py."""

from __future__ import annotations

import threading

import pytest

import app.metrics_collector as mc


class TestIncrement:
    def setup_method(self) -> None:
        mc.reset()

    def test_basic_increment(self) -> None:
        mc.increment("hits")
        snap = mc.snapshot()
        assert snap["counters"]["hits"] == 1

    def test_increment_by_n(self) -> None:
        mc.increment("reqs", amount=5)
        snap = mc.snapshot()
        assert snap["counters"]["reqs"] == 5

    def test_multiple_increments_accumulate(self) -> None:
        for _ in range(10):
            mc.increment("tick")
        snap = mc.snapshot()
        assert snap["counters"]["tick"] == 10

    def test_thread_safety(self) -> None:
        threads = [threading.Thread(target=mc.increment, args=("c",)) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        snap = mc.snapshot()
        assert snap["counters"]["c"] == 50


class TestRecordTiming:
    def setup_method(self) -> None:
        mc.reset()

    def test_timing_stored(self) -> None:
        mc.record_timing("predict", 25.0)
        snap = mc.snapshot()
        assert "predict" in snap["timings"]

    def test_timing_count(self) -> None:
        for i in range(5):
            mc.record_timing("api", float(i * 10))
        snap = mc.snapshot()
        assert snap["timings"]["api"]["count"] == 5

    def test_percentile_p95_gte_p50(self) -> None:
        for i in range(1, 101):
            mc.record_timing("lat", float(i))
        snap = mc.snapshot()
        t = snap["timings"]["lat"]
        assert t["p95_ms"] >= t["p50_ms"]


class TestSnapshot:
    def setup_method(self) -> None:
        mc.reset()

    def test_empty_snapshot(self) -> None:
        snap = mc.snapshot()
        assert snap["counters"] == {}
        assert snap["timings"] == {}

    def test_snapshot_has_both_keys(self) -> None:
        snap = mc.snapshot()
        assert "counters" in snap
        assert "timings" in snap


class TestReset:
    def test_clears_counters(self) -> None:
        mc.increment("x")
        mc.reset()
        assert mc.snapshot()["counters"] == {}

    def test_clears_timings(self) -> None:
        mc.record_timing("y", 10.0)
        mc.reset()
        assert mc.snapshot()["timings"] == {}
