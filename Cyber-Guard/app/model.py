"""ML model training and prediction for network intrusion detection."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from app.features import NetworkFeatureEngineer, build_feature_pipeline

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
METRICS_PATH = os.getenv("METRICS_PATH", "metrics.json")

THREAT_CLASSES = [
    "normal",
    "dos",
    "probe",
    "r2l",
    "u2r",
]

label_encoder = LabelEncoder()
label_encoder.fit(THREAT_CLASSES)


def _build_ensemble() -> VotingClassifier:
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
    )
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    return VotingClassifier(
        estimators=[("xgb", xgb), ("rf", rf)],
        voting="soft",
    )


def generate_synthetic_data(n_samples: int = 500) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    data = {
        "src_bytes": rng.exponential(500, n_samples),
        "dst_bytes": rng.exponential(300, n_samples),
        "duration": rng.exponential(2, n_samples),
        "protocol_type": rng.choice(["tcp", "udp", "icmp"], n_samples),
        "service": rng.choice(["http", "ftp", "smtp", "ssh", "dns", "other"], n_samples),
        "flag": rng.choice(["SF", "S0", "REJ", "RSTO", "OTH"], n_samples),
    }
    df = pd.DataFrame(data)
    labels = rng.choice(THREAT_CLASSES, n_samples, p=[0.6, 0.15, 0.1, 0.1, 0.05])
    return df, pd.Series(labels)


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_path: str = MODEL_PATH,
    metrics_path: str = METRICS_PATH,
) -> tuple[Pipeline, dict[str, Any]]:
    feature_pipe = build_feature_pipeline()
    ensemble = _build_ensemble()

    pipe = Pipeline([
        ("features", feature_pipe),
        ("model", ensemble),
    ])

    y_enc = label_encoder.transform(y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y_enc, cv=cv, scoring="accuracy")

    pipe.fit(X, y_enc)

    metrics: dict[str, Any] = {
        "accuracy_mean": float(cv_scores.mean()),
        "accuracy_std": float(cv_scores.std()),
        "n_features": 15,
        "n_samples": len(X),
        "classes": THREAT_CLASSES,
    }

    joblib.dump({"pipeline": pipe, "label_encoder": label_encoder}, model_path)
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    return pipe, metrics


def load_model(model_path: str = MODEL_PATH) -> tuple[Pipeline, LabelEncoder]:
    artifact = joblib.load(model_path)
    return artifact["pipeline"], artifact["label_encoder"]


def predict(
    X: pd.DataFrame,
    pipeline: Pipeline,
    le: LabelEncoder,
) -> dict[str, Any]:
    proba = pipeline.predict_proba(X)[0]
    class_idx = int(np.argmax(proba))
    label = le.inverse_transform([class_idx])[0]
    return {
        "prediction": label,
        "confidence": float(proba[class_idx]),
        "class_probabilities": {cls: float(p) for cls, p in zip(le.classes_, proba)},
    }


def ensure_model_exists(model_path: str = MODEL_PATH) -> None:
    if not Path(model_path).exists():
        X, y = generate_synthetic_data(500)
        train_model(X, y, model_path)
