"""Drift detection, prediction logging, and system metrics for Temporal-Pulse."""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

_REFERENCE_DISTRIBUTIONS: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
_CURRENT_DISTRIBUTIONS: dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
_PREDICTION_LOG: list[dict[str, Any]] = []
_REQUEST_TIMES: deque = deque(maxlen=1000)
DRIFT_ALPHA = 0.05


def update_reference_distribution(feature_name: str, values: list[float]) -> None:
    """Push values into the reference (training-time) distribution."""
    _REFERENCE_DISTRIBUTIONS[feature_name].extend(values)


def update_current_distribution(feature_name: str, values: list[float]) -> None:
    """Push values into the current (serving-time) distribution."""
    _CURRENT_DISTRIBUTIONS[feature_name].extend(values)


def run_ks_drift_test(feature_name: str) -> dict[str, Any]:
    """Run KS two-sample test between reference and current distributions.

    Returns drift detected flag, KS statistic, and p-value.
    """
    ref = list(_REFERENCE_DISTRIBUTIONS[feature_name])
    cur = list(_CURRENT_DISTRIBUTIONS[feature_name])

    if len(ref) < 30 or len(cur) < 30:
        return {"feature": feature_name, "drift_detected": False, "reason": "insufficient_data"}

    ks_stat, p_value = stats.ks_2samp(ref, cur)
    drift_detected = bool(p_value < DRIFT_ALPHA)
    result: dict[str, Any] = {
        "feature": feature_name,
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
        "drift_detected": drift_detected,
        "reference_mean": float(np.mean(ref)),
        "current_mean": float(np.mean(cur)),
        "reference_std": float(np.std(ref)),
        "current_std": float(np.std(cur)),
    }
    if drift_detected:
        logger.warning(
            "Data drift detected for feature '%s': KS=%.4f p=%.4f",
            feature_name,
            ks_stat,
            p_value,
        )
    return result


def run_all_drift_tests() -> list[dict[str, Any]]:
    """Run KS drift test for every tracked feature."""
    features = set(_REFERENCE_DISTRIBUTIONS) | set(_CURRENT_DISTRIBUTIONS)
    return [run_ks_drift_test(f) for f in features]


def log_prediction(
    sensor_id: str,
    anomaly_score: float,
    forecast: list[float],
    latency_ms: float,
    model_version: str = "v1.0",
) -> None:
    """Append a prediction event to the in-memory log."""
    _PREDICTION_LOG.append(
        {
            "sensor_id": sensor_id,
            "anomaly_score": anomaly_score,
            "forecast_steps": len(forecast),
            "latency_ms": latency_ms,
            "model_version": model_version,
            "timestamp": time.time(),
        }
    )
    _REQUEST_TIMES.append(latency_ms)


def get_metrics_summary() -> dict[str, Any]:
    """Return aggregated prediction and latency metrics."""
    if not _PREDICTION_LOG:
        return {"total_predictions": 0}

    scores = [e["anomaly_score"] for e in _PREDICTION_LOG]
    latencies = list(_REQUEST_TIMES)
    return {
        "total_predictions": len(_PREDICTION_LOG),
        "anomaly_score_mean": float(np.mean(scores)),
        "anomaly_score_p95": float(np.percentile(scores, 95)),
        "anomaly_score_p99": float(np.percentile(scores, 99)),
        "latency_mean_ms": float(np.mean(latencies)) if latencies else 0.0,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "drift_features_tested": len(_REFERENCE_DISTRIBUTIONS),
    }


def reset_monitoring() -> None:
    """Clear all in-memory monitoring state (used in tests)."""
    _REFERENCE_DISTRIBUTIONS.clear()
    _CURRENT_DISTRIBUTIONS.clear()
    _PREDICTION_LOG.clear()
    _REQUEST_TIMES.clear()
