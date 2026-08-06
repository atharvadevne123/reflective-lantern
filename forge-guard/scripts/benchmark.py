"""Latency benchmark for the Forge-Guard inference path.

Measures p50/p95/p99 latency of feature engineering + model prediction
without the HTTP layer, to validate the sub-50ms inference budget.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Any

from app.features import engineer_single
from app.model import load_model, predict

SAMPLE = {
    "temperature": 78.5,
    "pressure": 52.0,
    "vibration": 2.1,
    "cycle_time": 28.0,
    "tool_wear": 15.0,
    "power_consumption": 98.0,
    "humidity": 45.0,
}

SLA_BUDGET_MS = 50.0


def run_benchmark(n_iterations: int = 200, warmup: int = 10) -> dict[str, Any]:
    """Run n inference iterations and return latency percentiles in ms.

    Args:
        n_iterations: Number of timed inference calls.
        warmup: Number of untimed warm-up calls before measurement starts.

    Returns:
        Dict with p50, p95, p99, mean latencies plus pass/fail SLA flag.
    """
    model = load_model()

    for _ in range(warmup):
        predict(model, engineer_single(SAMPLE))

    latencies: list[float] = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        feats = engineer_single(SAMPLE)
        predict(model, feats)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()
    p50 = round(statistics.median(latencies), 2)
    p95 = round(latencies[int(0.95 * len(latencies))], 2)
    p99 = round(latencies[int(0.99 * len(latencies))], 2)
    mean = round(statistics.mean(latencies), 2)
    stdev = round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0.0

    return {
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "mean_ms": mean,
        "stdev_ms": stdev,
        "iterations": n_iterations,
        "sla_budget_ms": SLA_BUDGET_MS,
        "sla_pass": p95 <= SLA_BUDGET_MS,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forge-Guard inference latency benchmark")
    parser.add_argument("--iterations", type=int, default=200, help="Number of timed iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    results = run_benchmark(n_iterations=args.iterations, warmup=args.warmup)
    print("Forge-Guard inference latency benchmark")
    print("-" * 40)
    for key, value in results.items():
        print(f"  {key}: {value}")
    print("-" * 40)
    if results["sla_pass"]:
        print("SLA PASS — p95 within 50ms budget")
    else:
        print(f"SLA FAIL — p95 {results['p95_ms']}ms exceeds {SLA_BUDGET_MS}ms budget")
        sys.exit(1)
