"""ML model training, prediction, and persistence.

Supports XGBoost, LightGBM, and RandomForest via a VotingRegressor ensemble.
Models are serialised with joblib and metrics written to JSON.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor, VotingRegressor
from sklearn.model_selection import KFold, cross_val_score
from typing import Any

try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from app.features import build_feature_pipeline

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model.joblib"))
ANOMALY_MODEL_PATH = Path(os.getenv("ANOMALY_MODEL_PATH", "anomaly_model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "metrics.json"))

if not _HAS_XGB:
    logger.warning("xgboost not available; using RandomForest only")


def _build_estimator() -> VotingRegressor:
    """Build the voting ensemble from available backends."""
    estimators: list[tuple[str, Any]] = [
        ("rf", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)),
    ]
    if _HAS_XGB:
        estimators.append(
            ("xgb", XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0))
        )
    if _HAS_LGBM:
        estimators.append(
            ("lgbm", LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1))
        )
    return VotingRegressor(estimators=estimators)


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple[Any, dict[str, float]]:
    """Train the forecasting model with 5-fold CV; return fitted pipeline and metrics."""
    feature_pipe = build_feature_pipeline()
    X_feat = feature_pipe.fit_transform(X)

    estimator = _build_estimator()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(estimator, X_feat, y, cv=kf, scoring="r2")
    estimator.fit(X_feat, y)

    # Compute in-sample MAE
    preds = estimator.predict(X_feat)
    mae = float(np.mean(np.abs(preds - y.values)))

    metrics: dict[str, float] = {
        "r2_mean": float(cv_scores.mean()),
        "r2_std": float(cv_scores.std()),
        "mae_kwh": mae,
        "n_samples": len(y),
        "n_features": X_feat.shape[1],
    }

    joblib.dump({"pipeline": feature_pipe, "estimator": estimator}, MODEL_PATH)
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)

    logger.info("Model trained: R2=%.4f±%.4f  MAE=%.2f kWh", metrics["r2_mean"], metrics["r2_std"], mae)
    return {"pipeline": feature_pipe, "estimator": estimator}, metrics


def train_anomaly_model(X: pd.DataFrame) -> Any:
    """Train an IsolationForest on historical readings for anomaly detection."""
    feature_pipe = build_feature_pipeline()
    X_feat = feature_pipe.fit_transform(X)
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso.fit(X_feat)
    joblib.dump({"pipeline": feature_pipe, "iso": iso}, ANOMALY_MODEL_PATH)
    logger.info("Anomaly model trained on %d samples", len(X))
    return {"pipeline": feature_pipe, "iso": iso}


def load_model() -> dict[str, Any] | None:
    """Load the forecasting model bundle or return None."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def load_anomaly_model() -> dict[str, Any] | None:
    """Load the anomaly detection bundle or return None."""
    if not ANOMALY_MODEL_PATH.exists():
        return None
    return joblib.load(ANOMALY_MODEL_PATH)


def predict(model_bundle: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """Run forecasting model on feature DataFrame."""
    feat = model_bundle["pipeline"].transform(X)
    return model_bundle["estimator"].predict(feat)


def score_anomaly(anomaly_bundle: dict[str, Any], X: pd.DataFrame) -> dict[str, Any]:
    """Return anomaly score and flag for a reading."""
    feat = anomaly_bundle["pipeline"].transform(X)
    score = float(anomaly_bundle["iso"].score_samples(feat)[0])
    is_anomaly = int(anomaly_bundle["iso"].predict(feat)[0] == -1)
    severity = "none"
    if is_anomaly:
        severity = "critical" if score < -0.5 else "warning"
    return {"anomaly_score": round(score, 4), "is_anomaly": is_anomaly, "severity": severity}


def get_metrics() -> dict[str, float]:
    """Load latest training metrics."""
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH) as fh:
        return json.load(fh)
