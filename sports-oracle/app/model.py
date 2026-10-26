"""ML model training, evaluation, and prediction for Sports-Oracle."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.features import LABEL_ENCODER, build_feature_pipeline

logger = logging.getLogger(__name__)

MODEL_PATH = Path("model.joblib")
METRICS_PATH = Path("metrics.json")

OUTCOME_LABELS = ["A", "D", "H"]  # alphabetical — sklearn encodes this way


def _build_classifier() -> VotingClassifier:
    """Assemble XGBoost + LightGBM + RandomForest soft-voting ensemble."""
    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="mlogloss",
        random_state=42,
    )
    lgbm = LGBMClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
    )
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    return VotingClassifier(
        estimators=[("xgb", xgb), ("lgbm", lgbm), ("rf", rf)],
        voting="soft",
        weights=[2, 2, 1],
    )


def build_model_pipeline() -> Pipeline:
    """Return the full sklearn pipeline: feature eng → scaler → ensemble."""
    feature_pipe = build_feature_pipeline()
    classifier = _build_classifier()
    return Pipeline(
        [
            ("features", feature_pipe),
            ("scaler", StandardScaler()),
            ("model", classifier),
        ]
    )


def train_model(
    X: pd.DataFrame,
    y: np.ndarray,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> tuple[Pipeline, dict]:
    """Train the model pipeline and persist artefacts."""
    pipe = build_model_pipeline()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc_ovr_weighted")
    acc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")

    pipe.fit(X, y)

    metrics = {
        "run_id": str(uuid.uuid4()),
        "auc_mean": round(float(auc_scores.mean()), 4),
        "auc_std": round(float(auc_scores.std()), 4),
        "accuracy_mean": round(float(acc_scores.mean()), 4),
        "accuracy_std": round(float(acc_scores.std()), 4),
        "n_features": int(X.shape[1]),
        "n_samples": int(len(y)),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    metrics_path.write_text(json.dumps(metrics))

    logger.info(
        "model_trained",
        extra={"auc_mean": metrics["auc_mean"], "accuracy_mean": metrics["accuracy_mean"]},
    )
    return pipe, metrics


def load_model(model_path: Path = MODEL_PATH) -> Pipeline:
    """Load a persisted model pipeline."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


def read_metrics(metrics_path: Path = METRICS_PATH) -> dict:
    """Read persisted training metrics."""
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text())


def predict(
    pipe: Pipeline,
    X: pd.DataFrame,
) -> dict:
    """Return outcome probabilities and predicted label for one row."""
    probs = pipe.predict_proba(X)[0]
    class_order = LABEL_ENCODER.classes_  # ['A', 'D', 'H']
    prob_map = dict(zip(class_order.tolist(), [round(float(p), 4) for p in probs]))
    predicted_idx = int(np.argmax(probs))
    predicted_outcome = class_order[predicted_idx]
    confidence = round(float(probs[predicted_idx]), 4)
    return {
        "predicted_outcome": predicted_outcome,
        "prob_home": prob_map.get("H", 0.0),
        "prob_draw": prob_map.get("D", 0.0),
        "prob_away": prob_map.get("A", 0.0),
        "confidence": confidence,
    }


__all__ = [
    "build_model_pipeline",
    "train_model",
    "load_model",
    "read_metrics",
    "predict",
    "MODEL_PATH",
    "METRICS_PATH",
    "OUTCOME_LABELS",
]
