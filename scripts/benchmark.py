"""Measure inference latency across the feature pipeline and ensemble."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import statistics
import time

import pandas as pd

from app.features import build_feature_pipeline, generate_synthetic_data, prepare_X
from app.model import train_model

N_WARMUP = 20
N_RUNS = 200


def main() -> None:
    df = generate_synthetic_data(n=1500, seed=11)
    feat_pipe = build_feature_pipeline()
    X = prepare_X(df, feat_pipe, fit=True)
    y = df["delivery_minutes"].values

    t0 = time.perf_counter()
    model, metrics = train_model(X, y)
    train_s = time.perf_counter() - t0

    row = pd.DataFrame([{
        "carrier": "DHL", "distance_km": 42.5, "weight_kg": 3.2,
        "route_type": "urban", "hour_of_day": 14, "day_of_week": 2,
    }])

    for _ in range(N_WARMUP):
        model.predict(prepare_X(row, feat_pipe))

    latencies: list[float] = []
    for _ in range(N_RUNS):
        t = time.perf_counter()
        model.predict(prepare_X(row, feat_pipe))
        latencies.append((time.perf_counter() - t) * 1000)

    latencies.sort()
    print(f"Training       : {train_s:.2f} s on {len(df)} rows")
    print(f"CV RMSE        : {metrics['rmse_mean']:.2f} (±{metrics['rmse_std']:.2f})")
    print(f"CV R²          : {metrics['r2_mean']:.4f}")
    print(f"Latency mean   : {statistics.mean(latencies):.2f} ms")
    print(f"Latency p50    : {latencies[len(latencies) // 2]:.2f} ms")
    print(f"Latency p95    : {latencies[int(len(latencies) * 0.95)]:.2f} ms")
    print(f"Latency p99    : {latencies[int(len(latencies) * 0.99)]:.2f} ms")


if __name__ == "__main__":
    main()
