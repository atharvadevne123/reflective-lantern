"""ML model training, persistence, and inference for Ticket-Oracle.

Two models are trained:
  - priority_model: XGBoost + LightGBM + RandomForest VotingClassifier (P1-P4)
  - sla_model:      XGBoost binary classifier (SLA breach yes/no)
Both share the same ColumnTransformer feature pipeline.
"""

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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from app.features import make_feature_pipeline

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
PRIORITY_MODEL_PATH = MODEL_DIR / "priority_model.joblib"
SLA_MODEL_PATH = MODEL_DIR / "sla_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

PRIORITY_LABELS = ["P1", "P2", "P3", "P4"]


def _build_priority_pipeline() -> Pipeline:
    """Assemble the priority-classification pipeline.

    Returns:
        Unfitted sklearn Pipeline with ColumnTransformer + VotingClassifier.
    """
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="mlogloss",
        use_label_encoder=False,
        verbosity=0,
        random_state=42,
    )
    lgbm = LGBMClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        verbose=-1,
        random_state=42,
    )
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    voter = VotingClassifier(
        estimators=[("xgb", xgb), ("lgbm", lgbm), ("rf", rf)],
        voting="soft",
    )
    return Pipeline([
        ("features", make_feature_pipeline()),
        ("classifier", voter),
    ])


def _build_sla_pipeline() -> Pipeline:
    """Assemble the SLA breach binary-classification pipeline.

    Returns:
        Unfitted sklearn Pipeline with ColumnTransformer + XGBClassifier.
    """
    return Pipeline([
        ("features", make_feature_pipeline()),
        ("classifier", XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
            random_state=42,
        )),
    ])


def train_models(
    X: pd.DataFrame,
    y_priority: np.ndarray,
    y_sla: np.ndarray,
    model_version: str = "1.0.0",
) -> dict[str, Any]:
    """Train both models with 5-fold CV and persist them to MODEL_DIR.

    Args:
        X: Feature DataFrame from make_synthetic_dataset or real data.
        y_priority: Integer priority labels (0=P1 ... 3=P4).
        y_sla: Binary SLA breach labels.
        model_version: Semantic version string for metrics.json.

    Returns:
        Metrics dictionary with CV AUC-ROC for both models.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    priority_pipe = _build_priority_pipeline()
    priority_cv = cross_val_score(
        priority_pipe, X, y_priority, cv=cv, scoring="roc_auc_ovr_weighted"
    )
    logger.info("priority_cv_complete", extra={"auc_mean": float(priority_cv.mean()), "auc_std": float(priority_cv.std())})
    priority_pipe.fit(X, y_priority)

    sla_pipe = _build_sla_pipeline()
    sla_cv = cross_val_score(
        sla_pipe, X, y_sla, cv=cv, scoring="roc_auc"
    )
    logger.info("sla_cv_complete", extra={"auc_mean": float(sla_cv.mean()), "auc_std": float(sla_cv.std())})
    sla_pipe.fit(X, y_sla)

    metrics: dict[str, Any] = {
        "model_version": model_version,
        "priority_auc_mean": round(float(priority_cv.mean()), 4),
        "priority_auc_std": round(float(priority_cv.std()), 4),
        "sla_auc_mean": round(float(sla_cv.mean()), 4),
        "sla_auc_std": round(float(sla_cv.std()), 4),
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
    }

    joblib.dump(priority_pipe, PRIORITY_MODEL_PATH)
    joblib.dump(sla_pipe, SLA_MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("models_trained_and_saved", extra=metrics)
    return metrics


def load_models() -> tuple[Pipeline, Pipeline]:
    """Load persisted priority and SLA models from MODEL_DIR.

    Returns:
        Tuple of (priority_pipeline, sla_pipeline).

    Raises:
        FileNotFoundError: If model files are absent (not yet trained).
    """
    if not PRIORITY_MODEL_PATH.exists() or not SLA_MODEL_PATH.exists():
        raise FileNotFoundError("Models not found. Run train_models() first.")
    priority_pipe: Pipeline = joblib.load(PRIORITY_MODEL_PATH)
    sla_pipe: Pipeline = joblib.load(SLA_MODEL_PATH)
    logger.info("models_loaded")
    return priority_pipe, sla_pipe


def predict(
    priority_pipe: Pipeline,
    sla_pipe: Pipeline,
    X: pd.DataFrame,
) -> dict[str, Any]:
    """Run inference and return structured prediction output.

    Args:
        priority_pipe: Fitted priority classification pipeline.
        sla_pipe: Fitted SLA breach classification pipeline.
        X: One-row DataFrame from payload_to_dataframe.

    Returns:
        Dictionary with priority, probabilities, SLA breach risk, and
        estimated resolution hours.
    """
    priority_probs: np.ndarray = priority_pipe.predict_proba(X)[0]
    sla_prob: float = float(sla_pipe.predict_proba(X)[0][1])

    priority_idx: int = int(np.argmax(priority_probs))
    priority_label = PRIORITY_LABELS[priority_idx]
    confidence = float(priority_probs[priority_idx])

    resolution_hours = _estimate_resolution_hours(priority_idx, sla_prob)

    return {
        "priority": priority_label,
        "priority_probabilities": {
            lbl: round(float(p), 4) for lbl, p in zip(PRIORITY_LABELS, priority_probs, strict=True)
        },
        "sla_breach_risk": round(sla_prob, 4),
        "sla_breach_predicted": sla_prob >= 0.5,
        "estimated_resolution_hours": round(resolution_hours, 1),
        "confidence": round(confidence, 4),
    }


def _estimate_resolution_hours(priority_idx: int, sla_breach_risk: float) -> float:
    """Derive expected resolution time from priority class and SLA risk.

    Args:
        priority_idx: 0=P1, 1=P2, 2=P3, 3=P4.
        sla_breach_risk: SLA breach probability from the SLA model.

    Returns:
        Expected hours to resolution.
    """
    base_hours = [1.0, 4.0, 24.0, 72.0]
    base = base_hours[priority_idx]
    return base * (1.0 + sla_breach_risk * 0.5)


def read_champion_metrics() -> dict[str, Any]:
    """Read current champion metrics without side effects.

    Returns:
        Metrics dictionary, or empty dict if no metrics file exists.
    """
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)
