"""Ensemble ML model training, persistence, and inference."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

try:
    from lightgbm import LGBMRegressor

    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
METRICS_PATH = os.getenv("METRICS_PATH", "metrics.json")
MODEL_VERSION = "1.0.0"


def _build_ensemble() -> VotingRegressor:
    estimators: list[tuple[str, Any]] = [
        (
            "xgb",
            XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="rmse",
                verbosity=0,
            ),
        ),
        (
            "rf",
            RandomForestRegressor(
                n_estimators=150,
                max_depth=8,
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ]
    if _HAS_LGB:
        estimators.append(
            (
                "lgbm",
                LGBMRegressor(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    verbose=-1,
                ),
            )
        )
    return VotingRegressor(estimators=estimators)


def train_model(X: np.ndarray, y: np.ndarray) -> tuple[Pipeline, dict]:
    """Train ensemble, run 5-fold CV, persist model, return metrics."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", _build_ensemble()),
        ]
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    neg_rmse = cross_val_score(pipe, X, y, cv=kf, scoring="neg_root_mean_squared_error")
    r2_scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")

    pipe.fit(X, y)

    metrics = {
        "rmse_mean": round(float(-neg_rmse.mean()), 3),
        "rmse_std": round(float(neg_rmse.std()), 3),
        "r2_mean": round(float(r2_scores.mean()), 4),
        "r2_std": round(float(r2_scores.std()), 4),
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "model_version": MODEL_VERSION,
    }

    joblib.dump(pipe, MODEL_PATH)
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)

    logger.info(
        "Model trained — RMSE=%.2f (±%.2f), R²=%.4f",
        metrics["rmse_mean"],
        metrics["rmse_std"],
        metrics["r2_mean"],
    )
    return pipe, metrics


def load_model() -> Pipeline:
    """Load a persisted pipeline; train on synthetic data if not found."""
    if Path(MODEL_PATH).exists():
        return joblib.load(MODEL_PATH)

    logger.warning("Model file missing; training on synthetic data")
    from app.features import build_feature_pipeline, generate_synthetic_data, prepare_X

    df = generate_synthetic_data(n=2000)
    feat_pipe = build_feature_pipeline()
    X = prepare_X(df, feat_pipe, fit=True)
    y = df["delivery_minutes"].values
    pipe, _ = train_model(X, y)
    joblib.dump(feat_pipe, "feature_pipeline.joblib")
    return pipe


def predict(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Return predicted delivery times in minutes."""
    return model.predict(X)
