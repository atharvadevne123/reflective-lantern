"""Ensemble ML model training and prediction for energy load forecasting."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import KFold, cross_val_score

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "volt_cast_model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "volt_cast_metrics.json"))
MODEL_VERSION = "1.0.0"

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    logger.warning("xgboost not available; using RandomForest only")

try:
    from lightgbm import LGBMRegressor
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    logger.warning("lightgbm not available; using RandomForest only")


def _build_estimators() -> list:
    estimators = [
        ("rf", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)),
    ]
    if _HAS_XGB:
        estimators.append(
            ("xgb", XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                                  subsample=0.8, colsample_bytree=0.8,
                                  verbosity=0, random_state=42))
        )
    if _HAS_LGB:
        estimators.append(
            ("lgb", LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                                   subsample=0.8, colsample_bytree=0.8,
                                   verbose=-1, random_state=42))
        )
    return estimators


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
) -> tuple[VotingRegressor, dict]:
    estimators = _build_estimators()
    model = VotingRegressor(estimators=estimators)

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
    rmse_scores = cross_val_score(
        model, X, y, cv=kf,
        scoring="neg_root_mean_squared_error",
    )

    model.fit(X, y)

    metrics = {
        "r2_mean": float(r2_scores.mean()),
        "r2_std": float(r2_scores.std()),
        "rmse_mean": float(-rmse_scores.mean()),
        "rmse_std": float(rmse_scores.std()),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_estimators": len(estimators),
        "cv_folds": cv_folds,
        "model_version": MODEL_VERSION,
    }

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info("model trained: R2=%.4f RMSE=%.2f", metrics["r2_mean"], metrics["rmse_mean"])
    return model, metrics


def load_model() -> VotingRegressor | None:
    if not MODEL_PATH.exists():
        logger.warning("model file not found at %s", MODEL_PATH)
        return None
    model = joblib.load(MODEL_PATH)
    logger.info("model loaded from %s", MODEL_PATH)
    return model


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text())


def predict(model: VotingRegressor, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


def generate_synthetic_training_data(
    n_samples: int = 2000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    hours = rng.integers(0, 24, n_samples)
    dow = rng.integers(0, 7, n_samples)
    months = rng.integers(1, 13, n_samples)
    is_weekend = (dow >= 5).astype(int)
    temps = rng.uniform(-5, 40, n_samples)
    humidity = rng.uniform(20, 95, n_samples)

    base_load = 3500.0
    hour_effect = 500 * np.sin(np.pi * hours / 12) + 300 * (hours >= 7).astype(float)
    temp_effect = 100 * np.maximum(temps - 22, 0) + 80 * np.maximum(10 - temps, 0)
    weekend_effect = -400 * is_weekend
    month_effect = 200 * np.sin(2 * np.pi * months / 12)
    noise = rng.normal(0, 150, n_samples)

    load = base_load + hour_effect + temp_effect + weekend_effect + month_effect + noise
    load = np.clip(load, 1000, 8000)

    df = pd.DataFrame({
        "hour": hours,
        "hour_sin": np.sin(2 * np.pi * hours / 24),
        "hour_cos": np.cos(2 * np.pi * hours / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * months / 12),
        "month_cos": np.cos(2 * np.pi * months / 12),
        "day_of_week": dow,
        "month": months,
        "is_weekend": is_weekend,
        "is_peak_hour": ((hours >= 7) & (hours <= 21)).astype(int),
        "is_morning_peak": ((hours >= 7) & (hours <= 10)).astype(int),
        "is_evening_peak": ((hours >= 17) & (hours <= 21)).astype(int),
        "is_weekday_peak": (((hours >= 7) & (hours <= 21)) & (is_weekend == 0)).astype(int),
        "temperature_c": temps,
        "humidity_pct": humidity,
        "heat_index": temps + 0.33 * (humidity / 100.0 * 6.105) - 4.0,
        "cooling_demand_proxy": np.maximum(temps - 18.0, 0),
        "heating_demand_proxy": np.maximum(18.0 - temps, 0),
        "lag_1h": load * rng.uniform(0.95, 1.05, n_samples),
        "lag_24h": load * rng.uniform(0.90, 1.10, n_samples),
        "lag_168h": load * rng.uniform(0.85, 1.15, n_samples),
        "rolling_mean_3h": load * rng.uniform(0.97, 1.03, n_samples),
        "rolling_std_3h": rng.uniform(50, 200, n_samples),
        "rolling_mean_24h": load * rng.uniform(0.93, 1.07, n_samples),
        "rolling_std_24h": rng.uniform(100, 400, n_samples),
        "load_ratio_vs_daily_mean": rng.uniform(0.7, 1.3, n_samples),
    })
    return df, pd.Series(load, name="load_mw")
