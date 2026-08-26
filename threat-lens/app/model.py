"""Ensemble ML model: XGBoost + LightGBM + RandomForest for intrusion detection."""

import importlib.util
import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

_HAS_LGB = importlib.util.find_spec("lightgbm") is not None
if not _HAS_LGB:
    logger.warning("LightGBM not available; ensemble will use XGBoost + RF only")

ATTACK_CLASSES: list[str] = ["normal", "dos", "probe", "r2l", "u2r"]
MODEL_PATH = Path(os.getenv("MODEL_PATH", "model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "metrics.json"))


def _build_estimators() -> list[tuple[str, Any]]:
    estimators: list[tuple[str, Any]] = [
        (
            "xgb",
            XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                use_label_encoder=False,
                verbosity=0,
                random_state=42,
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=5,
                random_state=42,
                # Base estimators stay single-threaded: cross_val_score already
                # parallelises across folds, and nesting the two deadlocks joblib.
                n_jobs=1,
            ),
        ),
    ]
    if _HAS_LGB:
        import lightgbm as lgb  # noqa: PLC0415
        estimators.append((
            "lgb",
            lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbose=-1,
            ),
        ))
    return estimators


def build_pipeline() -> Pipeline:
    """Return the full sklearn Pipeline (scaler + voting ensemble)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ensemble", VotingClassifier(
            estimators=_build_estimators(),
            voting="soft",
        )),
    ])


def train_model(X: np.ndarray, y: np.ndarray) -> tuple[Pipeline, dict[str, float]]:
    """Train the ensemble and persist model + metrics to disk.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Integer labels 0-4 corresponding to ATTACK_CLASSES.

    Returns:
        Fitted pipeline and a metrics dict with AUC mean/std.
    """
    logger.info("Training ensemble on %d samples, %d features", X.shape[0], X.shape[1])
    pipe = build_pipeline()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=1)

    pipe.fit(X, y)

    metrics = {
        "accuracy_mean": float(scores.mean()),
        "accuracy_std": float(scores.std()),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": len(ATTACK_CLASSES),
    }
    logger.info("CV accuracy: %.4f ± %.4f", scores.mean(), scores.std())

    joblib.dump(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info("Model saved to %s", MODEL_PATH)
    return pipe, metrics


def load_model() -> Pipeline:
    """Load the persisted model from disk, training from scratch if absent."""
    if MODEL_PATH.exists():
        logger.info("Loading model from %s", MODEL_PATH)
        return joblib.load(MODEL_PATH)
    logger.warning("No saved model found — training default model now")
    from app.features import generate_synthetic_dataset  # noqa: PLC0415
    X, y = generate_synthetic_dataset()
    pipe, _ = train_model(X, y)
    return pipe


def predict(pipe: Pipeline, features: np.ndarray) -> dict[str, Any]:
    """Run inference and return class label with confidence scores.

    Args:
        pipe: Fitted sklearn Pipeline.
        features: 2D array of shape (1, n_features).

    Returns:
        Dict with predicted_class, is_attack, confidence, and class_probabilities.
    """
    proba = pipe.predict_proba(features)[0]
    class_idx = int(np.argmax(proba))
    predicted_class = ATTACK_CLASSES[class_idx]
    return {
        "predicted_class": predicted_class,
        "is_attack": int(predicted_class != "normal"),
        "confidence": round(float(proba[class_idx]), 4),
        "class_probabilities": {
            cls: round(float(p), 4) for cls, p in zip(ATTACK_CLASSES, proba)
        },
    }


def load_metrics() -> dict[str, Any]:
    """Return last saved training metrics or an empty dict."""
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}
