"""Measure inference latency across the feature pipeline and ensemble."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import statistics
import time

import pandas as pd

from app.features import build_feature_pipeline, generate_synthetic_data, prepare_X
from app.model import train_model

logger = logging.getLogger(__name__)

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
    logger.info("Training       : %.2f s on %d rows", train_s, len(df))
    logger.info("CV RMSE        : %.2f (±%.2f)", metrics["rmse_mean"], metrics["rmse_std"])
    logger.info("CV R²          : %.4f", metrics["r2_mean"])
    logger.info("Latency mean   : %.2f ms", statistics.mean(latencies))
    logger.info("Latency p50    : %.2f ms", latencies[len(latencies) // 2])
    logger.info("Latency p95    : %.2f ms", latencies[int(len(latencies) * 0.95)])
    logger.info("Latency p99    : %.2f ms", latencies[int(len(latencies) * 0.99)])


if __name__ == "__main__":
    main()
