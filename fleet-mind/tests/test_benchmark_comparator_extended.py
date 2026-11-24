"""Extended benchmark comparator tests with parametrize and boundary coverage."""
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


def test_benchmark_result_delta_positive() -> None:
    result = BenchmarkResult(metric="uptime", actual=0.99, benchmark=0.95)
    assert result.delta == pytest.approx(0.04)


def test_benchmark_result_delta_negative() -> None:
    result = BenchmarkResult(metric="failure_rate", actual=0.10, benchmark=0.05)
    assert result.delta == pytest.approx(0.05)


def test_benchmark_result_delta_pct_above() -> None:
    result = BenchmarkResult(metric="uptime", actual=1.1, benchmark=1.0)
    assert result.delta_pct == pytest.approx(10.0)


def test_benchmark_result_delta_pct_zero_benchmark() -> None:
    result = BenchmarkResult(metric="x", actual=5.0, benchmark=0.0)
    assert result.delta_pct == pytest.approx(0.0)


def test_benchmark_result_is_above_true() -> None:
    result = BenchmarkResult(metric="uptime", actual=0.99, benchmark=0.95)
    assert result.is_above_benchmark is True


def test_benchmark_result_is_above_false() -> None:
    result = BenchmarkResult(metric="uptime", actual=0.90, benchmark=0.95)
    assert result.is_above_benchmark is False


def test_compare_returns_benchmark_result() -> None:
    result = compare("fuel_efficiency", 35.0, 30.0, unit="mpg")
    assert isinstance(result, BenchmarkResult)
    assert result.metric == "fuel_efficiency"
    assert result.unit == "mpg"


@pytest.mark.parametrize("actual,benchmark,expected_above", [
    (100.0, 90.0, True),
    (90.0, 100.0, False),
    (100.0, 100.0, False),
])
def test_compare_is_above_benchmark_parametrized(
    actual: float, benchmark: float, expected_above: bool
) -> None:
    result = compare("metric", actual, benchmark)
    assert result.is_above_benchmark is expected_above


def test_fleet_vs_industry_returns_common_metrics() -> None:
    fleet = {"uptime": 0.99, "fuel_efficiency": 35.0, "mtbf": 1200.0}
    industry = {"uptime": 0.95, "fuel_efficiency": 30.0}
    results = fleet_vs_industry(fleet, industry)
    assert len(results) == 2


def test_fleet_vs_industry_empty_fleet() -> None:
    results = fleet_vs_industry({}, {"uptime": 0.95})
    assert results == []


def test_fleet_vs_industry_no_common_metrics() -> None:
    results = fleet_vs_industry({"x": 1.0}, {"y": 2.0})
    assert results == []


def test_below_benchmark_metrics_filters() -> None:
    results = [
        BenchmarkResult("a", 0.80, 0.90),
        BenchmarkResult("b", 0.95, 0.90),
    ]
    below = below_benchmark_metrics(results)
    assert len(below) == 1
    assert below[0].metric == "a"


def test_above_benchmark_metrics_filters() -> None:
    results = [
        BenchmarkResult("a", 0.80, 0.90),
        BenchmarkResult("b", 0.95, 0.90),
    ]
    above = above_benchmark_metrics(results)
    assert len(above) == 1
    assert above[0].metric == "b"


def test_mean_delta_pct_empty_returns_zero() -> None:
    assert mean_delta_pct([]) == pytest.approx(0.0)


def test_mean_delta_pct_mixed() -> None:
    results = [
        BenchmarkResult("a", 1.1, 1.0),
        BenchmarkResult("b", 0.9, 1.0),
    ]
    result = mean_delta_pct(results)
    assert isinstance(result, float)


@pytest.mark.parametrize("n", [1, 3, 5, 10])
def test_fleet_vs_industry_n_matching_metrics(n: int) -> None:
    keys = [f"metric_{i}" for i in range(n)]
    fleet = {k: float(i + 1) for i, k in enumerate(keys)}
    industry = {k: float(i) for i, k in enumerate(keys)}
    results = fleet_vs_industry(fleet, industry)
    assert len(results) == n
