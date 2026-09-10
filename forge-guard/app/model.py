"""Ensemble ML model training, persistence, and inference for Forge-Guard."""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
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

# sklearn >=1.9 uses get_tags().__sklearn_tags__().estimator_type to validate classifiers.
# XGBClassifier 2.1.x doesn't set estimator_type in its __sklearn_tags__, causing
# VotingClassifier to reject it.  Patch at class level so clones inherit the fix.
_orig_xgb_tags = XGBClassifier.__sklearn_tags__  # type: ignore[attr-defined]


def _patched_xgb_tags(self) -> object:  # type: ignore[no-untyped-def]
    tags = _orig_xgb_tags(self)
    tags.estimator_type = "classifier"
    return tags


XGBClassifier.__sklearn_tags__ = _patched_xgb_tags  # type: ignore[method-assign]

from app.features import generate_synthetic_data  # noqa: E402

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
    # n_jobs=1: parallelised cross-val spawns subprocesses that miss the
    # XGBClassifier._estimator_type patch for sklearn >=1.9 compatibility.
    cv_scores = cross_val_score(full_pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
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

    try:
        from app.mlflow_compat import log_run

        log_run(metrics, params={"cv_folds": cv_folds, "model_version": MODEL_VERSION})
    except Exception:
        logger.debug("Experiment tracking unavailable — continuing.")

    try:
        from app.faiss_index import build_index

        healthy = X[y == 0]
        if len(healthy) >= 50:
            build_index(healthy.astype(np.float32))
    except Exception:
        logger.debug("FAISS index build skipped.")

    return full_pipeline, metrics


def load_model() -> Pipeline:
    """Load the trained model pipeline from disk, training if absent.

    Falls back to synthetic-data training when no model file exists.
    Raises RuntimeError if the file is present but cannot be deserialised.
    """
    if not MODEL_PATH.exists():
        logger.warning("No model found at %s — training on synthetic data.", MODEL_PATH)
        df = generate_synthetic_data()
        from app.features import build_feature_pipeline

        feat_pipe = build_feature_pipeline()
        feature_cols = [c for c in df.columns if c != "defect"]
        X = feat_pipe.fit_transform(df[feature_cols])
        y = df["defect"].values
        train_model(X, y)

    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        logger.error("Failed to load model from %s: %s", MODEL_PATH, exc)
        raise RuntimeError(f"Model load failed: {exc}") from exc


def predict(
    model: Pipeline,
    features: np.ndarray,
) -> tuple[int, float]:
    """Run inference and return (predicted_class, defect_probability)."""
    prob = model.predict_proba(features)[0, 1]
    label = int(prob >= 0.5)
    return label, round(float(prob), 4)


@lru_cache(maxsize=1)
def _read_metrics_cached(path_str: str) -> dict[str, Any]:
    """Read and cache metrics JSON from disk. Cache is invalidated on path change."""
    path = Path(path_str)
    if path.exists():
        return json.loads(path.read_text())
    return {}


def get_metrics() -> dict[str, Any]:
    """Return persisted training metrics or empty dict if none exist."""
    return _read_metrics_cached(str(METRICS_PATH))


def feature_importance(model: Pipeline) -> dict[str, float]:
    """Return a feature-importance dict from the XGBoost sub-estimator."""

    estimator = model.named_steps["model"]
    xgb_est = estimator.named_estimators_["xgb"]
    scores = xgb_est.feature_importances_
    from app.features import FEATURE_NAMES

    if FEATURE_NAMES and len(FEATURE_NAMES) == len(scores):
        return {name: round(float(s), 4) for name, s in zip(FEATURE_NAMES, scores, strict=False)}
    return {f"f{i}": round(float(s), 4) for i, s in enumerate(scores)}
