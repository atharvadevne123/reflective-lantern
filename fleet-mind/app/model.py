"""XGBoost + LightGBM ensemble model for fleet maintenance prediction."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import xgboost as xgb
    import lightgbm as lgb
    _HAS_BOOSTING = True
except ImportError:
    _HAS_BOOSTING = False

from app.config import get_settings
from app.features import ALL_FEATURES, build_feature_pipeline

logger = logging.getLogger(__name__)

MODEL_VERSION = "1.0.0"

ANOMALY_TYPES = {
    "engine_overheat": lambda r: r.get("engine_temp_c", 0) > 105,
    "oil_pressure_low": lambda r: r.get("oil_pressure_bar", 5) < 1.5,
    "brake_worn": lambda r: r.get("brake_pad_mm", 12) < 2.5,
    "battery_weak": lambda r: r.get("battery_voltage_v", 12.6) < 11.5,
    "tire_pressure_low": lambda r: r.get("tire_pressure_bar", 2.4) < 1.8,
    "transmission_overheat": lambda r: r.get("transmission_temp_c", 80) > 120,
}


def _build_estimator() -> Any:
    """Build the voting ensemble or a fallback GradientBoosting estimator."""
    if _HAS_BOOSTING:
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        return VotingClassifier(
            estimators=[("xgb", xgb_clf), ("lgb", lgb_clf)],
            voting="soft",
            n_jobs=-1,
        )
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)


def _generate_training_data(n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic fleet telemetry for initial training."""
    rng = np.random.default_rng(42)
    rows = []
    labels = []
    for _ in range(n):
        normal = rng.random() > 0.25
        row = {
            "speed_kmh": rng.uniform(0, 130),
            "engine_temp_c": rng.uniform(60, 95) if normal else rng.uniform(95, 130),
            "oil_pressure_bar": rng.uniform(2.5, 5.5) if normal else rng.uniform(0.5, 2.5),
            "fuel_consumption_lph": rng.uniform(4, 12),
            "battery_voltage_v": rng.uniform(12.0, 13.5) if normal else rng.uniform(10.5, 12.0),
            "brake_pad_mm": rng.uniform(3, 12) if normal else rng.uniform(0.5, 3),
            "tire_pressure_bar": rng.uniform(2.0, 2.8) if normal else rng.uniform(1.2, 2.0),
            "rpm": rng.uniform(700, 4500),
            "coolant_temp_c": rng.uniform(70, 90) if normal else rng.uniform(88, 110),
            "transmission_temp_c": rng.uniform(60, 90) if normal else rng.uniform(90, 140),
        }
        rows.append(row)
        labels.append(0 if normal else 1)

    import pandas as pd
    df = pd.DataFrame(rows)
    pipeline = build_feature_pipeline()
    X = pipeline.fit_transform(df)
    return X, np.array(labels)


def train_model() -> dict[str, float]:
    """Train the ensemble, evaluate with 5-fold CV, and persist to disk."""
    settings = get_settings()
    logger.info("Training Fleet-Mind model …")

    X, y = _generate_training_data(2000)
    estimator = _build_estimator()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(estimator, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    estimator.fit(X, y)
    y_pred = estimator.predict(X)
    y_prob = estimator.predict_proba(X)[:, 1]

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "cv_auc_mean": float(cv_auc.mean()),
        "cv_auc_std": float(cv_auc.std()),
        "train_samples": int(len(y)),
    }

    Path(settings.model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, settings.model_path)

    Path(settings.metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(settings.metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    logger.info("Model trained: AUC=%.4f CV-AUC=%.4f±%.4f", metrics["roc_auc"], metrics["cv_auc_mean"], metrics["cv_auc_std"])
    return metrics


def load_model() -> Any:
    """Load persisted model, training if not found."""
    settings = get_settings()
    if not Path(settings.model_path).exists():
        logger.info("No saved model found — training from scratch")
        train_model()
    return joblib.load(settings.model_path)


def get_metrics() -> dict[str, float]:
    """Return last recorded model metrics."""
    settings = get_settings()
    try:
        with open(settings.metrics_path) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def predict(model: Any, features: np.ndarray, raw_reading: dict[str, float]) -> dict[str, Any]:
    """Run inference and return structured prediction."""
    prob = float(model.predict_proba(features)[0, 1])
    maintenance_needed = prob >= 0.5

    anomaly_type = None
    for name, check in ANOMALY_TYPES.items():
        if check(raw_reading):
            anomaly_type = name
            break

    # Heuristic remaining useful life based on brake pad thickness
    brake_mm = raw_reading.get("brake_pad_mm", 12.0)
    rul = max(0.0, (brake_mm - 2.0) / 10.0 * 50_000)

    return {
        "maintenance_needed": maintenance_needed,
        "anomaly_score": round(prob, 4),
        "anomaly_type": anomaly_type,
        "remaining_useful_life_km": round(rul, 1),
        "confidence": round(abs(prob - 0.5) * 2, 4),
        "model_version": MODEL_VERSION,
    }
