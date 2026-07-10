"""Anomaly detection for energy load readings using Isolation Forest."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

_iso_model: IsolationForest | None = None


def fit_isolation_forest(
    loads: list[float],
    contamination: float = 0.05,
    random_state: int = 42,
) -> IsolationForest:
    """Fit an Isolation Forest on the reference load distribution."""
    arr = np.array(loads, dtype=float).reshape(-1, 1)
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    model.fit(arr)
    logger.info("isolation forest fitted on %d samples", len(loads))
    return model


def score_loads(
    model: IsolationForest,
    loads: list[float],
) -> list[dict]:
    """Score each load value and flag anomalies (-1 = anomaly, 1 = normal)."""
    arr = np.array(loads, dtype=float).reshape(-1, 1)
    scores = model.decision_function(arr)
    labels = model.predict(arr)

    return [
        {
            "index": i,
            "load_mw": round(float(loads[i]), 2),
            "anomaly_score": round(float(scores[i]), 4),
            "is_anomaly": bool(labels[i] == -1),
        }
        for i in range(len(loads))
    ]


def quick_anomaly_check(
    reference_loads: list[float],
    new_loads: list[float],
) -> dict:
    """Fit on reference and score new_loads — returns summary."""
    if len(reference_loads) < 20:
        return {"error": "insufficient_reference_data"}

    model = fit_isolation_forest(reference_loads)
    results = score_loads(model, new_loads)
    n_anomalies = sum(1 for r in results if r["is_anomaly"])

    return {
        "total_checked": len(new_loads),
        "anomalies_detected": n_anomalies,
        "anomaly_rate_pct": round(100 * n_anomalies / max(len(new_loads), 1), 1),
        "details": results,
    }
