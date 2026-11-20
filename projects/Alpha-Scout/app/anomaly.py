"""Isolation Forest anomaly detection for feature-level outliers."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

_detector: IsolationForest | None = None


def fit_anomaly_detector(X: pd.DataFrame, contamination: float = 0.05) -> IsolationForest:
    """Fit an IsolationForest on training features."""
    global _detector
    _detector = IsolationForest(contamination=contamination, random_state=42, n_jobs=1)
    _detector.fit(X.fillna(0))
    logger.info("anomaly detector fitted n_samples=%d n_features=%d", len(X), X.shape[1])
    return _detector


def predict_anomaly(features: pd.DataFrame) -> dict[str, Any]:
    """Predict whether a feature row is anomalous."""
    if _detector is None:
        return {"is_anomaly": False, "anomaly_score": 0.0, "n_rows": len(features), "error": "not fitted"}
    scores = _detector.decision_function(features.fillna(0))
    preds = _detector.predict(features.fillna(0))
    is_anomaly = bool(np.any(preds == -1))
    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": round(float(np.min(scores)), 4),
        "n_rows": len(features),
    }
