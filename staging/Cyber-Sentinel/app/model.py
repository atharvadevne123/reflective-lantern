"""Ensemble ML model: XGBoost + LightGBM + RandomForest for intrusion detection."""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBClassifier

    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning("XGBoost not available; using RandomForest only")

try:
    from lightgbm import LGBMClassifier

    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

from app.exceptions import ModelNotTrainedError
from app.features import generate_synthetic_data

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/tmp/cyber_sentinel_model.pkl"))
LABEL_ENCODER_PATH = Path(
    os.environ.get("LABEL_ENCODER_PATH", "/tmp/cyber_sentinel_le.pkl")
)

ATTACK_TYPES = ["normal", "dos", "probe", "r2l", "u2r"]

_model_cache: dict[str, Any] = {}


def _build_estimators() -> list[tuple[str, Any]]:
    """Construct list of base estimators for the ensemble."""
    estimators: list[tuple[str, Any]] = [
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=-1, random_state=42)),
    ]
    if _XGB_AVAILABLE:
        estimators.append(
            (
                "xgb",
                XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    use_label_encoder=False,
                    eval_metric="mlogloss",
                    random_state=42,
                    verbosity=0,
                ),
            )
        )
    if _LGB_AVAILABLE:
        estimators.append(
            (
                "lgb",
                LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1,
                ),
            )
        )
    return estimators


def train_model(X: np.ndarray | None = None, y: np.ndarray | None = None) -> dict[str, Any]:
    """Train the ensemble classifier and persist it to disk.

    If X and y are None, synthetic data is generated for demonstration.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Integer label array of shape (n_samples,).

    Returns:
        Dictionary with model metadata: version, f1_score, cv_mean, report.
    """
    if X is None or y is None:
        logger.info("Generating synthetic training data")
        X, y = generate_synthetic_data(n_samples=2000)

    le = LabelEncoder()
    le.fit(list(range(5)))

    estimators = _build_estimators()
    voting_clf = VotingClassifier(estimators=estimators, voting="soft")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", voting_clf),
    ])

    logger.info("Starting cross-validation with %d samples", len(X))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
    logger.info("CV F1 scores: %s", cv_scores)

    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    f1 = float(f1_score(y, y_pred, average="macro"))
    report = classification_report(y, y_pred, output_dict=True)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    _model_cache["model"] = pipeline
    _model_cache["le"] = le
    _model_cache["version"] = "1.0.0"

    logger.info("Model trained and saved. F1=%.4f, CV mean=%.4f", f1, cv_scores.mean())
    metrics = {
        "version": "1.0.0",
        "f1_score": round(f1, 4),
        "cv_mean": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
        "report": report,
        "n_samples": len(X),
        "estimators": [e[0] for e in estimators],
    }
    # Persist metrics alongside the model so load_metrics() has data to serve.
    with open(MODEL_PATH.parent / "metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    return metrics


def _load_model() -> tuple[Pipeline, LabelEncoder]:
    """Load model from cache or disk."""
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["le"]
    if MODEL_PATH.exists() and LABEL_ENCODER_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(LABEL_ENCODER_PATH, "rb") as f:
            le = pickle.load(f)
        _model_cache["model"] = model
        _model_cache["le"] = le
        return model, le
    raise ModelNotTrainedError("Model not trained. Call /train first.")


def predict(features: list[float]) -> dict[str, Any]:
    """Run inference on a single feature vector.

    Args:
        features: Numeric feature vector produced by build_feature_vector().

    Returns:
        Dict with label, confidence, attack_type, and probabilities.
    """
    model, le = _load_model()
    X = np.array(features, dtype=float).reshape(1, -1)
    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    confidence = float(proba[pred_idx])
    label = "attack" if pred_idx > 0 else "normal"
    attack_type = ATTACK_TYPES[pred_idx]

    return {
        "label": label,
        "confidence": confidence,
        "attack_type": attack_type,
        "class_probabilities": {
            ATTACK_TYPES[i]: round(float(p), 4) for i, p in enumerate(proba)
        },
        "model_version": _model_cache.get("version", "unknown"),
    }


def get_feature_importance() -> dict[str, float]:
    """Return feature importances from the Random Forest base learner."""
    model, _ = _load_model()
    pipeline = model
    clf = pipeline.named_steps["clf"]
    # A fitted VotingClassifier exposes named_estimators_; its estimators_
    # list holds bare (unnamed) estimators and cannot be unpacked as pairs.
    rf = getattr(clf, "named_estimators_", {}).get("rf")
    if rf is None and hasattr(clf, "feature_importances_"):
        rf = clf
    if rf is not None and hasattr(rf, "feature_importances_"):
        importances = rf.feature_importances_
        return {f"feature_{i}": round(float(v), 6) for i, v in enumerate(importances)}
    return {}


def load_metrics() -> dict[str, Any]:
    """Return persisted training metrics if available."""
    metrics_path = MODEL_PATH.parent / "metrics.pkl"
    if metrics_path.exists():
        with open(metrics_path, "rb") as f:
            return pickle.load(f)
    return {"status": "no metrics available"}


def clear_model_cache() -> None:
    """Evict the in-memory model cache (useful for testing)."""
    _model_cache.clear()
