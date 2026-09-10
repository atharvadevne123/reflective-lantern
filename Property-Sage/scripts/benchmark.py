"""Inference latency benchmarking script for Property-Sage.

Measures the mean and p95 prediction latency for a batch of
synthetic property records using the loaded ensemble model.

Usage:
    PYTHONPATH=. python scripts/benchmark.py [n_requests]
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING)

from app.features import generate_synthetic_data, property_to_dataframe  # noqa: E402
from app.model import load_models, predict  # noqa: E402


def benchmark(n: int = 500) -> dict[str, float]:
    """Run n inference calls and report latency statistics.

    Args:
        n: Number of predictions to benchmark.

    Returns:
        Dict with mean, median, p95, p99, and min latency in milliseconds.
    """
    price_model, rental_model = load_models()
    X, _, _ = generate_synthetic_data(n=n, seed=123)
    latencies = []

    for _, row in X.iterrows():
        df = property_to_dataframe(row.to_dict())
        t0 = time.perf_counter()
        predict(price_model, rental_model, df)
        latencies.append((time.perf_counter() - t0) * 1000)

    stats = {
        "n": n,
        "mean_ms": round(float(np.mean(latencies)), 3),
        "median_ms": round(float(np.median(latencies)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies, 99)), 3),
        "min_ms": round(float(np.min(latencies)), 3),
        "max_ms": round(float(np.max(latencies)), 3),
    }
    return stats


if __name__ == "__main__":
    n_requests = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    print("Running benchmark...")
    results = benchmark(n_requests)
    for k, v in results.items():
        print(f"  {k}: {v}")
