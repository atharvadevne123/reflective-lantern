"""ML model training and prediction for ticket triage."""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from app.features import build_feature_pipeline

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/priority_model.joblib")
ENCODER_PATH = Path("models/priority_encoder.joblib")
REGRESSOR_PATH = Path("models/resolution_model.joblib")
METRICS_PATH = Path("models/metrics.json")

PRIORITY_WEIGHTS = {"critical": 4, "high": 3, "medium": 2, "low": 1}


def build_ensemble() -> VotingClassifier:
    """Build XGBoost + LightGBM + RandomForest soft-voting ensemble."""
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        verbosity=0,
    )
    lgbm = LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
    )
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=-1, random_state=42)
    return VotingClassifier(
        estimators=[("xgb", xgb), ("lgbm", lgbm), ("rf", rf)],
        voting="soft",
    )


def train_model(df: pd.DataFrame) -> tuple[Pipeline, dict[str, Any]]:
    """Train priority classifier and resolution time regressor."""
    Path("models").mkdir(exist_ok=True)

    le = LabelEncoder()
    y_priority = le.fit_transform(df["priority"])

    feature_pipe = build_feature_pipeline()
    X = feature_pipe.fit_transform(df[["subject", "description"]])

    ensemble = build_ensemble()
    cv_scores = cross_val_score(ensemble, X, y_priority, cv=5, scoring="roc_auc_ovr")

    # VotingClassifier requires metadata routing for per-estimator sample_weight;
    # fit without it to stay compatible with sklearn < 1.4 metadata routing defaults.
    ensemble.fit(X, y_priority)

    # Build full pipeline object (feature extraction + classifier)
    full_pipe = Pipeline(
        [
            ("features", feature_pipe),
            ("clf", ensemble),
        ]
    )

    # Also train a resolution-time regressor (LightGBM regression)
    from lightgbm import LGBMRegressor

    reg_pipe = Pipeline(
        [
            ("features", build_feature_pipeline()),
            ("reg", LGBMRegressor(n_estimators=100, verbose=-1)),
        ]
    )
    reg_pipe.fit(df[["subject", "description"]], df["resolution_hours"])

    metrics = {
        "auc_mean": float(cv_scores.mean()),
        "auc_std": float(cv_scores.std()),
        "n_features": int(X.shape[1]),
        "n_samples": int(len(df)),
        "classes": list(le.classes_),
    }

    joblib.dump(full_pipe, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    joblib.dump(reg_pipe, REGRESSOR_PATH)
    json.dump(metrics, METRICS_PATH.open("w"), indent=2)

    logger.info(
        "Model trained: AUC=%.4f+-%.4f on %d samples", metrics["auc_mean"], metrics["auc_std"], metrics["n_samples"]
    )
    return full_pipe, metrics


def load_model() -> tuple[Pipeline, LabelEncoder, Any]:
    """Load trained model, encoder, and regressor from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run /train first.")
    pipe = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    reg = joblib.load(REGRESSOR_PATH)
    return pipe, le, reg


def predict(
    subject: str,
    description: str,
    model: Pipeline,
    encoder: LabelEncoder,
    regressor: Any,
) -> dict[str, Any]:
    """Predict ticket priority, category, and resolution hours."""
    df = pd.DataFrame([{"subject": subject, "description": description}])
    proba = model.predict_proba(df)[0]
    class_idx = int(np.argmax(proba))
    priority = encoder.classes_[class_idx]
    resolution_hours = float(regressor.predict(df)[0])

    return {
        "priority": priority,
        "confidence": float(proba[class_idx]),
        "priority_probabilities": {c: float(p) for c, p in zip(encoder.classes_, proba, strict=False)},
        "resolution_hours_estimate": max(0.5, round(resolution_hours, 1)),
    }
