"""Tests for fleet-mind app/benchmark_comparator.py."""

from __future__ import annotations

import pytest

from app.benchmark_comparator import (
    BenchmarkResult,
    above_benchmark_metrics,
    below_benchmark_metrics,
    compare,
    fleet_vs_industry,
    mean_delta_pct,
)


class TestBenchmarkResult:
    def test_delta_positive(self) -> None:
        r = BenchmarkResult("fuel", 12.0, 10.0)
        assert r.delta == pytest.approx(2.0)

    def test_delta_pct(self) -> None:
        r = BenchmarkResult("fuel", 12.0, 10.0)
        assert r.delta_pct == pytest.approx(20.0)

    def test_zero_benchmark_pct(self) -> None:
        r = BenchmarkResult("x", 5.0, 0.0)
        assert r.delta_pct == pytest.approx(0.0)


class TestCompare:
    def test_above_benchmark(self) -> None:
        r = compare("temp", 95.0, 90.0)
        assert r.is_above_benchmark is True

    def test_below_benchmark(self) -> None:
        r = compare("efficiency", 7.0, 9.0)
        assert r.is_above_benchmark is False


class TestFleetVsIndustry:
    def test_common_metrics_compared(self) -> None:
        fleet = {"fuel": 12.0, "uptime": 0.95}
        industry = {"fuel": 10.0, "uptime": 0.90}
        results = fleet_vs_industry(fleet, industry)
        assert len(results) == 2

    def test_missing_benchmark_excluded(self) -> None:
        fleet = {"fuel": 12.0, "unknown": 5.0}
        industry = {"fuel": 10.0}
        results = fleet_vs_industry(fleet, industry)
        assert len(results) == 1


class TestAboveAndBelowBenchmark:
    def test_above_returns_positives(self) -> None:
        results = [
            BenchmarkResult("a", 10.0, 8.0),
            BenchmarkResult("b", 7.0, 9.0),
        ]
        assert len(above_benchmark_metrics(results)) == 1
        assert len(below_benchmark_metrics(results)) == 1


class TestMeanDeltaPct:
    def test_empty_returns_zero(self) -> None:
        assert mean_delta_pct([]) == pytest.approx(0.0)

    def test_calculates_mean(self) -> None:
        results = [
            BenchmarkResult("a", 11.0, 10.0),
            BenchmarkResult("b", 9.0, 10.0),
        ]
        assert mean_delta_pct(results) == pytest.approx(0.0)
