"""ML model: Isolation Forest + XGBoost ensemble for anomaly scoring and forecasting."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/tmp/temporal_pulse_models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

_IF_MODEL: IsolationForest | None = None
_RF_MODEL: RandomForestRegressor | None = None
_SCALER: RobustScaler | None = None
_MODEL_VERSION: str = "untrained"


def train_anomaly_detector(
    X: np.ndarray,
    contamination: float = 0.05,
    n_estimators: int = 100,
    random_state: int = 42,
) -> tuple[IsolationForest, RobustScaler]:
    """Train Isolation Forest anomaly detector with robust scaling."""
    global _IF_MODEL, _SCALER, _MODEL_VERSION

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    _IF_MODEL = model
    _SCALER = scaler
    _MODEL_VERSION = "v1.0"
    logger.info("Isolation Forest trained on %d samples, %d features", *X.shape)
    return model, scaler


def train_forecaster(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 200,
    random_state: int = 42,
) -> tuple[RandomForestRegressor, dict[str, float]]:
    """Train Random Forest forecaster and return CV metrics."""
    global _RF_MODEL

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=12,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
    model.fit(X, y)

    _RF_MODEL = model
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "cv_mae_mean": float(-cv_scores.mean()),
        "cv_mae_std": float(cv_scores.std()),
    }
    logger.info("RF forecaster trained: MAE=%.4f RMSE=%.4f", mae, rmse)
    return model, metrics


def score_anomaly(
    X: np.ndarray,
    if_model: IsolationForest | None = None,
    scaler: RobustScaler | None = None,
) -> np.ndarray:
    """Return anomaly scores in [0, 1] where higher = more anomalous."""
    if_model = if_model or _IF_MODEL
    scaler = scaler or _SCALER
    if if_model is None or scaler is None:
        raise RuntimeError("Anomaly detector not trained. Call train_anomaly_detector first.")

    X_scaled = scaler.transform(X)
    raw_scores = if_model.score_samples(X_scaled)
    normalized = 1.0 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
    return normalized.astype(np.float32)


def predict_forecast(
    X: np.ndarray,
    horizon: int = 1,
    rf_model: RandomForestRegressor | None = None,
) -> list[float]:
    """Return point forecasts for the requested horizon."""
    rf_model = rf_model or _RF_MODEL
    if rf_model is None:
        raise RuntimeError("Forecaster not trained. Call train_forecaster first.")

    predictions = []
    x_input = X.copy()
    for _ in range(horizon):
        pred = float(rf_model.predict(x_input[-1:].reshape(1, -1))[0])
        predictions.append(pred)
        x_input = np.roll(x_input, -1, axis=0)
        x_input[-1] = x_input[-2]
    return predictions


def get_feature_importance(
    rf_model: RandomForestRegressor | None = None,
    feature_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return feature importances sorted descending."""
    rf_model = rf_model or _RF_MODEL
    if rf_model is None:
        return []

    importances = rf_model.feature_importances_
    names = feature_names or [f"feature_{i}" for i in range(len(importances))]
    ranked = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
    return [{"feature": n, "importance": float(v)} for n, v in ranked[:20]]


@lru_cache(maxsize=1)
def get_model_version() -> str:
    """Return the current model version string."""
    return _MODEL_VERSION
