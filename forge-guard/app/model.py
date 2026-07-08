"""Ensemble ML model training, persistence, and inference for Forge-Guard."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.features import generate_synthetic_data

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "metrics.json"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")


def _build_ensemble() -> VotingClassifier:
    """Construct the soft-voting XGBoost + RandomForest ensemble."""
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        
        random_state=42,
        n_jobs=-1,
    )
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=7,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    return VotingClassifier(estimators=[("xgb", xgb), ("rf", rf)], voting="soft")


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train the ensemble with 5-fold CV and persist artefacts.

    Returns the fitted pipeline and a metrics dict with AUC mean/std.
    """
    ensemble = _build_ensemble()
    full_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", ensemble),
        ]
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(full_pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    logger.info("CV AUC: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    full_pipeline.fit(X, y)

    y_prob = full_pipeline.predict_proba(X)[:, 1]
    train_auc = roc_auc_score(y, y_prob)

    metrics: dict[str, Any] = {
        "auc_cv_mean": round(float(cv_scores.mean()), 4),
        "auc_cv_std": round(float(cv_scores.std()), 4),
        "auc_train": round(float(train_auc), 4),
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "cv_folds": cv_folds,
        "model_version": MODEL_VERSION,
    }

    joblib.dump(full_pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info("Model saved to %s — metrics: %s", MODEL_PATH, metrics)
    return full_pipeline, metrics


def load_model() -> Pipeline:
    """Load the trained model pipeline from disk, training if absent."""
    if not MODEL_PATH.exists():
        logger.warning("No model found at %s — training on synthetic data.", MODEL_PATH)
        df = generate_synthetic_data()
        from app.features import build_feature_pipeline

        feat_pipe = build_feature_pipeline()
        feature_cols = [c for c in df.columns if c != "defect"]
        X = feat_pipe.fit_transform(df[feature_cols])
        y = df["defect"].values
        train_model(X, y)

    return joblib.load(MODEL_PATH)


def predict(
    model: Pipeline,
    features: np.ndarray,
) -> tuple[int, float]:
    """Run inference and return (predicted_class, defect_probability)."""
    prob = model.predict_proba(features)[0, 1]
    label = int(prob >= 0.5)
    return label, round(float(prob), 4)


def get_metrics() -> dict[str, Any]:
    """Return persisted training metrics or empty dict if none exist."""
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}
