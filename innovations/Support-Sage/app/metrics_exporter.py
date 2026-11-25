"""Prometheus metrics exporter for Support-Sage."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest
    _PROM_AVAILABLE = True
    PREDICT_COUNTER = Counter("support_sage_predictions_total", "Total predictions", ["priority"])
    PREDICT_LATENCY = Histogram("support_sage_prediction_latency_seconds", "Prediction latency")
    DRIFT_GAUGE = Gauge("support_sage_drift_detected", "1 if drift detected, 0 otherwise")
except ImportError:
    _PROM_AVAILABLE = False


def get_prometheus_metrics() -> bytes | str:
    """Return Prometheus text format metrics."""
    if not _PROM_AVAILABLE:
        return b"# prometheus_client not installed\n"
    return generate_latest()
