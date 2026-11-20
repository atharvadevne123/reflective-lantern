"""Benchmark comparison utilities for fleet performance metrics."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass
class BenchmarkResult:
    metric: str
    actual: float
    benchmark: float
    unit: str = ""

    @property
    def delta(self) -> float:
        return self.actual - self.benchmark

    @property
    def delta_pct(self) -> float:
        if self.benchmark == 0:
            return 0.0
        return (self.delta / abs(self.benchmark)) * 100.0

    @property
    def is_above_benchmark(self) -> bool:
        return self.actual > self.benchmark


def compare(metric: str, actual: float, benchmark: float, unit: str = "") -> BenchmarkResult:
    """Create a benchmark comparison result."""
    return BenchmarkResult(metric=metric, actual=actual, benchmark=benchmark, unit=unit)


def fleet_vs_industry(
    fleet_metrics: dict[str, float],
    industry_benchmarks: dict[str, float],
) -> list[BenchmarkResult]:
    """Return benchmark results for all metrics present in both dicts."""
    results = []
    for metric, actual in fleet_metrics.items():
        if metric in industry_benchmarks:
            results.append(compare(metric, actual, industry_benchmarks[metric]))
    return results


def below_benchmark_metrics(results: list[BenchmarkResult]) -> list[BenchmarkResult]:
    """Return metrics where actual is below benchmark (potential under-performance)."""
    return [r for r in results if not r.is_above_benchmark]


def above_benchmark_metrics(results: list[BenchmarkResult]) -> list[BenchmarkResult]:
    """Return metrics where actual exceeds benchmark."""
    return [r for r in results if r.is_above_benchmark]


def mean_delta_pct(results: list[BenchmarkResult]) -> float:
    """Return mean percentage delta across all benchmark results."""
    if not results:
        return 0.0
    return mean(r.delta_pct for r in results)
