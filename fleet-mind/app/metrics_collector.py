"""Prometheus-compatible metrics collection for Fleet-Mind.

When prometheus_client is installed, exposes Counter and Histogram metrics.
Otherwise, accumulates counts in memory for the /metrics endpoint fallback.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram  # type: ignore[import]
    _HAS_PROMETHEUS = True

    PREDICTIONS_TOTAL = Counter(
        "fleet_mind_predictions_total",
        "Total number of predictions served",
        ["result"],  # 'maintenance' | 'normal'
    )
    PREDICTION_LATENCY = Histogram(
        "fleet_mind_prediction_latency_seconds",
        "Prediction endpoint latency",
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    )
    DRIFT_DETECTIONS = Counter(
        "fleet_mind_drift_detections_total",
        "Total KS-test drift detections by feature",
        ["feature"],
    )
except ImportError:
    _HAS_PROMETHEUS = False

# In-memory fallback counters
_counters: dict[str, int] = defaultdict(int)
_latencies: list[float] = []


def record_prediction(maintenance_needed: bool, latency_s: float) -> None:
    """Record a served prediction.

    Args:
        maintenance_needed: Whether this prediction flagged maintenance.
        latency_s: End-to-end prediction latency in seconds.
    """
    label = "maintenance" if maintenance_needed else "normal"
    if _HAS_PROMETHEUS:
        PREDICTIONS_TOTAL.labels(result=label).inc()
        PREDICTION_LATENCY.observe(latency_s)
    else:
        _counters[f"predictions_{label}"] += 1
        _latencies.append(latency_s)
        if len(_latencies) > 10_000:
            _latencies.pop(0)


def record_drift(feature: str) -> None:
    """Increment the drift detection counter for a feature.

    Args:
        feature: Name of the feature for which drift was detected.
    """
    if _HAS_PROMETHEUS:
        DRIFT_DETECTIONS.labels(feature=feature).inc()
    else:
        _counters[f"drift_{feature}"] += 1


def get_in_memory_metrics() -> dict[str, Any]:
    """Return accumulated in-memory metrics for the /metrics fallback endpoint.

    Returns:
        Dict with prediction counts, average latency, and drift counts.
    """
    avg_latency = sum(_latencies) / len(_latencies) if _latencies else 0.0
    return {
        "predictions_maintenance": _counters.get("predictions_maintenance", 0),
        "predictions_normal": _counters.get("predictions_normal", 0),
        "avg_latency_ms": round(avg_latency * 1000, 2),
        "drift_counts": {k[6:]: v for k, v in _counters.items() if k.startswith("drift_")},
    }
