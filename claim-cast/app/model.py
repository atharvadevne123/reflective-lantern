"""XGBoost ensemble model training and prediction for claim severity."""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.features import ActuarialFeatureEngineer

MODEL_PATH = Path(os.getenv("MODEL_PATH", "claim_cast_model.joblib"))
ANOMALY_PATH = Path(os.getenv("ANOMALY_PATH", "claim_cast_anomaly.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "claim_cast_metrics.json"))

SEVERITY_LABELS = {0: "low", 1: "medium", 2: "high", 3: "critical"}


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("features", ActuarialFeatureEngineer()),
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
        )),
    ])


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict]:
    pipe = build_pipeline()
    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="f1_macro")
    pipe.fit(X, y)

    anomaly_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    feat_eng = ActuarialFeatureEngineer()
    X_feats = feat_eng.fit_transform(X)
    anomaly_model.fit(X_feats)

    metrics = {
        "f1_macro_mean": float(cv_scores.mean()),
        "f1_macro_std": float(cv_scores.std()),
        "n_features": X.shape[1],
        "n_samples": len(X),
        "severity_classes": list(SEVERITY_LABELS.values()),
    }

    joblib.dump(pipe, MODEL_PATH)
    joblib.dump(anomaly_model, ANOMALY_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    return pipe, metrics


def load_model() -> tuple[Pipeline, IsolationForest]:
    if not MODEL_PATH.exists() or not ANOMALY_PATH.exists():
        _seed_model()
    return joblib.load(MODEL_PATH), joblib.load(ANOMALY_PATH)


def _seed_model() -> None:
    """Seed with synthetic data so the API starts even without real training data."""
    rng = np.random.default_rng(42)
    n = 500
    X = pd.DataFrame({
        "claim_age_days": rng.integers(0, 400, n).astype(float),
        "policy_limit": rng.choice([50_000, 100_000, 250_000, 500_000], n).astype(float),
        "claimed_amount": rng.exponential(25_000, n),
        "num_prior_claims": rng.integers(0, 8, n).astype(float),
        "policy_tenure_years": rng.uniform(0.5, 20, n),
        "geo_risk_tier": rng.integers(1, 6, n).astype(float),
        "days_since_last_claim": rng.integers(10, 1000, n).astype(float),
        "policy_type": rng.choice(["home", "auto", "life", "commercial"], n),
    })
    ratio = (X["claimed_amount"] / X["policy_limit"]).clip(0, 1.0)
    y = pd.cut(
        ratio,
        bins=[-0.001, 0.1, 0.3, 0.6, 1.001],
        labels=[0, 1, 2, 3],
    ).astype(int)
    train_model(X, y)


def predict(X: pd.DataFrame) -> dict:
    pipe, anomaly_model = load_model()
    proba = pipe.predict_proba(X)[0]
    severity_class = int(np.argmax(proba))
    severity_score = float(proba[severity_class])

    feat_eng = ActuarialFeatureEngineer()
    X_feats = feat_eng.fit_transform(X)
    anomaly_flag = int(anomaly_model.predict(X_feats)[0] == -1)

    triage_priority = _compute_triage(severity_class, anomaly_flag)

    return {
        "severity_class": severity_class,
        "severity_label": SEVERITY_LABELS[severity_class],
        "severity_score": round(severity_score, 4),
        "probabilities": {SEVERITY_LABELS[i]: round(float(p), 4) for i, p in enumerate(proba)},
        "anomaly_flag": anomaly_flag,
        "triage_priority": triage_priority,
    }


def _compute_triage(severity_class: int, anomaly_flag: int) -> int:
    """P1=critical/anomaly, P2=high, P3=medium, P4=low."""
    if severity_class == 3 or anomaly_flag:
        return 1
    if severity_class == 2:
        return 2
    if severity_class == 1:
        return 3
    return 4
