"""ML model: XGBoost + LightGBM + RandomForest VotingRegressor ensemble."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

from app.features import build_feature_pipeline, prepare_dataframe

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "metrics.json"))


def build_ensemble() -> VotingRegressor:
    xgb = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="rmse",
        verbosity=0,
    )
    lgbm = LGBMRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
    )
    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    return VotingRegressor([("xgb", xgb), ("lgbm", lgbm), ("rf", rf)])


def generate_synthetic_data(n: int = 2000) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    hours = np.arange(n) % 24
    days = (np.arange(n) // 24) % 7
    base = 3.0 + 2.0 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 0.3, n)
    weekend_factor = np.where(days >= 5, 0.8, 1.0)
    temp = 15.0 + 10.0 * np.sin(2 * np.pi * np.arange(n) / (24 * 365)) + rng.normal(0, 2, n)
    cooling = np.where(temp > 22, (temp - 22) * 0.3, 0.0)
    heating = np.where(temp < 10, (10 - temp) * 0.2, 0.0)
    consumption = (base * weekend_factor + cooling + heating).clip(min=0.1)
    df = pd.DataFrame({
        "consumption_kwh": consumption,
        "temperature_c": temp,
        "humidity_pct": 50.0 + rng.normal(0, 10, n),
        "hour_of_day": hours,
        "day_of_week": days,
        "is_holiday": (rng.random(n) < 0.03).astype(int),
        "building_type": rng.choice(["residential", "commercial", "office"], n),
    })
    return df, pd.Series(consumption)


def train_model(
    X_raw: pd.DataFrame | None = None,
    y: pd.Series | None = None,
) -> tuple[Any, dict[str, float]]:
    if X_raw is None or y is None:
        logger.info("No data provided — generating synthetic training data")
        X_raw, y = generate_synthetic_data()

    pipeline = build_feature_pipeline()
    X = pipeline.fit_transform(X_raw)
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.nan_to_num(X, nan=0.0)

    ensemble = build_ensemble()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = cross_val_score(ensemble, X, y, cv=cv, scoring="r2")
    rmse_scores = cross_val_score(ensemble, X, y, cv=cv, scoring="neg_root_mean_squared_error")

    ensemble.fit(X, y)

    metrics = {
        "r2_mean": float(r2_scores.mean()),
        "r2_std": float(r2_scores.std()),
        "rmse_mean": float(-rmse_scores.mean()),
        "rmse_std": float(rmse_scores.std()),
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
    }

    joblib.dump({"pipeline": pipeline, "model": ensemble}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    logger.info("Model trained: R2=%.4f RMSE=%.4f", metrics["r2_mean"], metrics["rmse_mean"])
    return {"pipeline": pipeline, "model": ensemble}, metrics


def load_model() -> dict[str, Any]:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    logger.warning("No model found — training on synthetic data")
    bundle, _ = train_model()
    return bundle


def predict(
    bundle: dict[str, Any],
    readings: list[dict],
    horizon_h: int = 1,
) -> list[float]:
    df = prepare_dataframe(readings)
    X = bundle["pipeline"].transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.nan_to_num(X, nan=0.0)
    preds = bundle["model"].predict(X)
    return [float(max(0.0, p * horizon_h)) for p in preds]


def get_metrics() -> dict[str, float]:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


def log_training_event(metrics: dict[str, float], model_path: str = str(MODEL_PATH)) -> None:
    """Emit a structured log line for model training completion."""
    logger.info(
        "model_trained r2=%.4f rmse=%.4f n_features=%d path=%s",
        metrics.get("r2_mean", 0),
        metrics.get("rmse_mean", 0),
        metrics.get("n_features", 0),
        model_path,
    )
