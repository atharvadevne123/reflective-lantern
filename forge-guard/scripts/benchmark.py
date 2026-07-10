"""Latency benchmark for the Forge-Guard inference path.

Measures p50/p95/p99 latency of feature engineering + model prediction
without the HTTP layer, to validate the sub-50ms inference budget.
"""

from __future__ import annotations

import statistics
import time

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


def run_benchmark(n_iterations: int = 200) -> dict[str, float]:
    """Run n inference iterations and return latency percentiles in ms."""
    model = load_model()

    # Warm-up
    for _ in range(10):
        predict(model, engineer_single(SAMPLE))

    latencies: list[float] = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        feats = engineer_single(SAMPLE)
        predict(model, feats)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()
    return {
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(latencies[int(0.95 * len(latencies))], 2),
        "p99_ms": round(latencies[int(0.99 * len(latencies))], 2),
        "mean_ms": round(statistics.mean(latencies), 2),
        "iterations": n_iterations,
    }


if __name__ == "__main__":
    results = run_benchmark()
    print("Forge-Guard inference latency benchmark")
    for key, value in results.items():
        print(f"  {key}: {value}")
