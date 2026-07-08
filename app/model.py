"""ML model training and ensemble prediction for traffic congestion level."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.features import FEATURE_COLUMNS, build_feature_dataframe

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "metrics.json"))
MODEL_VERSION = "1.0.0"

CONGESTION_LABELS: dict[int, str] = {
    0: "free",
    1: "moderate",
    2: "congested",
    3: "severe",
}


def _xgb_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    eval_metric="mlogloss",
                    random_state=42,
                    verbosity=0,
                ),
            ),
        ]
    )


def _lgbm_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LGBMClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42,
                    verbose=-1,
                ),
            ),
        ]
    )


def generate_synthetic_data(n_samples: int = 5000) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate labelled synthetic traffic records for training."""
    rng = np.random.default_rng(42)
    records: list[dict[str, Any]] = []
    labels: list[int] = []

    for _ in range(n_samples):
        hour = int(rng.integers(0, 24))
        dow = int(rng.integers(0, 7))
        month = int(rng.integers(1, 13))
        vehicle_count = int(rng.integers(100, 3000))
        avg_speed = max(5.0, float(rng.normal(60, 20)))
        incident_count = int(rng.poisson(0.5))
        temperature = float(rng.normal(18, 8))
        is_raining = int(rng.random() < 0.15)

        score = (
            (vehicle_count / 3000) * 3.0
            + (1 - avg_speed / 100) * 2.0
            + incident_count * 0.5
            + int(7 <= hour <= 9 or 16 <= hour <= 19) * 0.8
            + is_raining * 0.3
            - int(dow >= 5) * 0.4
            + float(rng.normal(0, 0.2))
        )
        level = min(3, max(0, int(score)))

        records.append(
            {
                "hour": hour,
                "day_of_week": dow,
                "month": month,
                "vehicle_count": vehicle_count,
                "avg_speed_kmh": avg_speed,
                "incident_count": incident_count,
                "temperature_celsius": temperature,
                "is_raining": is_raining,
                "road_type": "arterial",
            }
        )
        labels.append(level)

    return build_feature_dataframe(records), np.array(labels)


def train_model(
    X: pd.DataFrame | None = None,
    y: np.ndarray | None = None,
) -> tuple[dict[str, Pipeline], dict[str, Any]]:
    """Train XGBoost and LightGBM pipelines with 5-fold cross-validation."""
    if X is None or y is None:
        X, y = generate_synthetic_data()

    X_arr = X[FEATURE_COLUMNS].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models: dict[str, Pipeline] = {}
    cv_results: dict[str, Any] = {}

    for name, pipe in [("xgb", _xgb_pipeline()), ("lgbm", _lgbm_pipeline())]:
        scores = cross_val_score(pipe, X_arr, y, cv=cv, scoring="roc_auc_ovr_weighted")
        pipe.fit(X_arr, y)
        models[name] = pipe
        cv_results[name] = {"auc_mean": float(scores.mean()), "auc_std": float(scores.std())}
        logger.info("Trained %s AUC=%.4f±%.4f", name, scores.mean(), scores.std())

    metrics: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "n_samples": int(len(y)),
        "n_features": int(X_arr.shape[1]),
        "cv_results": cv_results,
        "ensemble_auc": float(np.mean([v["auc_mean"] for v in cv_results.values()])),
    }
    joblib.dump(models, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info("Model saved to %s metrics=%s", MODEL_PATH, metrics)
    return models, metrics


def load_model() -> dict[str, Pipeline]:
    if not MODEL_PATH.exists():
        logger.info("No saved model found — training from scratch")
        models, _ = train_model()
        return models
    return joblib.load(MODEL_PATH)  # type: ignore[no-any-return]


def predict(
    models: dict[str, Pipeline],
    features: pd.DataFrame,
) -> dict[str, Any]:
    """Average class probabilities from all ensemble members and return prediction."""
    X = features[FEATURE_COLUMNS].values
    probs: np.ndarray = np.mean([m.predict_proba(X) for m in models.values()], axis=0)
    congestion_level = int(np.argmax(probs, axis=1)[0])
    congestion_prob = float(probs[0][congestion_level])
    incident_score = float(
        features["incident_count"].iloc[0] / max(1.0, features["vehicle_count"].iloc[0]) * 100
    )
    return {
        "congestion_level": congestion_level,
        "congestion_label": CONGESTION_LABELS[congestion_level],
        "congestion_probability": round(congestion_prob, 4),
        "class_probabilities": {
            CONGESTION_LABELS[i]: round(float(p), 4) for i, p in enumerate(probs[0])
        },
        "incident_score": round(incident_score, 4),
        "model_version": MODEL_VERSION,
    }
