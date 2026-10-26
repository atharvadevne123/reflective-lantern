"""ML model training, prediction, and persistence for Sport-Cast."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from app.features import make_feature_pipeline, make_synthetic_dataset

logger = logging.getLogger(__name__)

MODEL_PATH = Path("model.joblib")
PIPELINE_PATH = Path("pipeline.joblib")
METRICS_PATH = Path("metrics.json")
CHAMPION_METRICS_PATH = Path("champion_metrics.json")

OUTCOME_LABELS = {0: "away_win", 1: "draw", 2: "home_win"}
OUTCOME_INDICES = {v: k for k, v in OUTCOME_LABELS.items()}


def build_ensemble() -> VotingClassifier:
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        use_label_encoder=False,
        verbosity=0,
    )
    lgb_params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
    }
    try:
        from lightgbm import LGBMClassifier
        lgb = LGBMClassifier(**lgb_params)
    except ImportError:
        lgb = XGBClassifier(n_estimators=150, max_depth=3, eval_metric="mlogloss",
                            use_label_encoder=False, verbosity=0)
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    return VotingClassifier(
        estimators=[("xgb", xgb), ("lgb", lgb), ("rf", rf)],
        voting="soft",
        weights=[0.5, 0.3, 0.2],
    )


def read_champion_auc() -> float:
    if CHAMPION_METRICS_PATH.exists():
        try:
            return float(json.loads(CHAMPION_METRICS_PATH.read_text()).get("auc_mean", 0.0))
        except Exception:
            return 0.0
    return 0.0


def train_model(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cv_folds: int = 5,
    save: bool = True,
) -> tuple[VotingClassifier, object, dict]:
    champion_auc = read_champion_auc()

    pipeline = make_feature_pipeline()
    X_transformed = pipeline.fit_transform(X)

    ensemble = build_ensemble()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(ensemble, X_transformed, y, cv=cv, scoring="roc_auc_ovr")

    metrics: dict = {
        "auc_mean": round(float(scores.mean()), 4),
        "auc_std": round(float(scores.std()), 4),
        "n_features": int(X_transformed.shape[1]),
        "n_samples": int(len(y)),
        "cv_folds": cv_folds,
    }

    if metrics["auc_mean"] >= champion_auc:
        ensemble.fit(X_transformed, y)
        if save:
            joblib.dump(ensemble, MODEL_PATH)
            joblib.dump(pipeline, PIPELINE_PATH)
            METRICS_PATH.write_text(json.dumps(metrics))
            CHAMPION_METRICS_PATH.write_text(json.dumps(metrics))
        logger.info("Model promoted: auc=%.4f (champion=%.4f)", metrics["auc_mean"], champion_auc)
    else:
        logger.warning("Challenger rejected: auc=%.4f < champion=%.4f", metrics["auc_mean"], champion_auc)
        if CHAMPION_METRICS_PATH.exists():
            METRICS_PATH.write_text(CHAMPION_METRICS_PATH.read_text())

    return ensemble, pipeline, metrics


def load_model() -> tuple[VotingClassifier, object]:
    if not MODEL_PATH.exists() or not PIPELINE_PATH.exists():
        logger.info("No saved model found — training on synthetic data")
        X, y = make_synthetic_dataset()
        train_model(X, y)
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    return model, pipeline


def predict_match(
    features: pd.DataFrame,
    model: VotingClassifier,
    pipeline: object,
) -> dict:
    X = pipeline.transform(features)
    proba = model.predict_proba(X)[0]
    predicted_idx = int(np.argmax(proba))
    return {
        "predicted_outcome": OUTCOME_LABELS[predicted_idx],
        "home_win_prob": round(float(proba[2]), 4),
        "draw_prob": round(float(proba[1]), 4),
        "away_win_prob": round(float(proba[0]), 4),
        "confidence": round(float(proba[predicted_idx]), 4),
    }


def score_player_performance(
    goals_avg: float,
    assists_avg: float,
    win_rate: float,
    minutes_played_ratio: float,
    injury_days_out: int,
) -> dict:
    attack = min(goals_avg * 20 + assists_avg * 10, 40)
    availability = minutes_played_ratio * 30
    form = win_rate * 20
    fitness = max(0.0, 10.0 - injury_days_out * 0.5)
    total = attack + availability + form + fitness
    fatigue_idx = float(np.exp(-minutes_played_ratio * 0.5))
    return {
        "performance_score": round(min(total, 100.0), 2),
        "fatigue_index": round(fatigue_idx, 4),
        "form_rating": round(form / 20.0, 4),
    }
