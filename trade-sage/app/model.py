"""ML model training, prediction, and persistence for Trade-Sage."""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = "model.joblib"
_DEFAULT_METRICS_PATH = "metrics.json"


def _model_path() -> str:
    return os.getenv("MODEL_PATH", _DEFAULT_MODEL_PATH)


def _metrics_path() -> str:
    return os.getenv("METRICS_PATH", _DEFAULT_METRICS_PATH)


def _build_ensemble() -> Any:
    try:
        from lightgbm import LGBMClassifier
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        )
        lgbm = LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            n_jobs=1,
        )
        return VotingClassifier(
            estimators=[("xgb", xgb), ("lgbm", lgbm), ("rf", rf)],
            voting="soft",
        )
    except ImportError:
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            RandomForestClassifier,
            VotingClassifier,
        )

        gb = GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)
        rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=1)
        return VotingClassifier(estimators=[("gb", gb), ("rf", rf)], voting="soft")


def train_model(X: np.ndarray, y: np.ndarray) -> tuple[Pipeline, dict]:
    """Train ensemble pipeline with 5-fold CV and persist artifacts."""
    run_id = str(uuid.uuid4())[:8]
    logger.info("Training run %s — %d samples, %d features", run_id, len(X), X.shape[1])

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", _build_ensemble()),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1)

    pipe.fit(X, y)

    metrics = {
        "run_id": run_id,
        "auc_mean": float(np.mean(auc_scores)),
        "auc_std": float(np.std(auc_scores)),
        "n_features": int(X.shape[1]),
        "n_samples": len(X),
    }

    joblib.dump(pipe, _model_path())
    with open(_metrics_path(), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Training complete — AUC %.4f ± %.4f", metrics["auc_mean"], metrics["auc_std"])
    return pipe, metrics


def load_model() -> Pipeline:
    path = Path(_model_path())
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {_model_path()}. Train first.")
    return joblib.load(path)


def predict(pipe: Pipeline, X: np.ndarray) -> tuple[list[int], list[float]]:
    """Return class predictions and confidence scores."""
    probs = pipe.predict_proba(X)
    preds = probs.argmax(axis=1).tolist()
    confs = probs.max(axis=1).tolist()
    return preds, confs


def model_is_loaded() -> bool:
    """Return True when a saved model artifact exists on disk."""
    return Path(_model_path()).exists()


def safe_predict(pipe: Pipeline, X: np.ndarray) -> tuple[list[int], list[float]] | None:
    """Wrapper around predict that returns None on any exception.

    Args:
        pipe: Trained sklearn Pipeline.
        X: Feature matrix of shape (n_samples, n_features).

    Returns:
        (predictions, confidences) tuple, or None if inference fails.
    """
    try:
        return predict(pipe, X)
    except Exception:
        logger.exception("safe_predict failed")
        return None


def read_champion_auc() -> float:
    if not Path(_metrics_path()).exists():
        return 0.0
    with open(_metrics_path()) as f:
        data = json.load(f)
    return float(data.get("auc_mean", 0.0))
