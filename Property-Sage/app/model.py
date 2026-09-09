"""ML model training, evaluation, and inference for Property-Sage."""

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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from app.features import PropertyFeatureEngineer, generate_synthetic_data

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
PRICE_MODEL_PATH = MODEL_DIR / "price_model.joblib"
RENTAL_MODEL_PATH = MODEL_DIR / "rental_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


def _build_ensemble() -> VotingRegressor:
    xgb = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="rmse",
        random_state=42,
        verbosity=0,
    )
    lgbm = LGBMRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    return VotingRegressor(
        estimators=[("xgb", xgb), ("lgbm", lgbm), ("rf", rf)],
        weights=[0.4, 0.4, 0.2],
    )


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("features", PropertyFeatureEngineer()),
        ("scaler", StandardScaler()),
        ("model", _build_ensemble()),
    ])


def train_model(
    X: pd.DataFrame,
    y_price: pd.Series,
    y_rental: pd.Series,
) -> dict[str, Any]:
    MODEL_DIR.mkdir(exist_ok=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    logger.info("Training price model with 5-fold CV...")
    price_pipe = _build_pipeline()
    price_cv = cross_val_score(price_pipe, X, y_price, cv=kf, scoring="r2", n_jobs=-1)
    price_pipe.fit(X, y_price)
    price_pred = price_pipe.predict(X)
    price_rmse = float(np.sqrt(mean_squared_error(y_price, price_pred)))
    price_mae = float(mean_absolute_error(y_price, price_pred))

    logger.info("Training rental yield model with 5-fold CV...")
    rental_pipe = _build_pipeline()
    rental_cv = cross_val_score(rental_pipe, X, y_rental, cv=kf, scoring="r2", n_jobs=-1)
    rental_pipe.fit(X, y_rental)
    rental_pred = rental_pipe.predict(X)
    rental_rmse = float(np.sqrt(mean_squared_error(y_rental, rental_pred)))
    rental_mae = float(mean_absolute_error(y_rental, rental_pred))

    joblib.dump(price_pipe, PRICE_MODEL_PATH)
    joblib.dump(rental_pipe, RENTAL_MODEL_PATH)

    metrics = {
        "price_r2_mean": float(price_cv.mean()),
        "price_r2_std": float(price_cv.std()),
        "price_rmse": price_rmse,
        "price_mae": price_mae,
        "rental_r2_mean": float(rental_cv.mean()),
        "rental_r2_std": float(rental_cv.std()),
        "rental_rmse": rental_rmse,
        "rental_mae": rental_mae,
        "n_train": len(X),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info("Models saved. Price R2=%.4f, Rental R2=%.4f", price_cv.mean(), rental_cv.mean())
    return metrics


def load_models() -> tuple[Pipeline, Pipeline]:
    if not PRICE_MODEL_PATH.exists() or not RENTAL_MODEL_PATH.exists():
        logger.info("Models not found — training on synthetic data")
        X, y_price, y_rental = generate_synthetic_data(n=2000)
        train_model(X, y_price, y_rental)
    return joblib.load(PRICE_MODEL_PATH), joblib.load(RENTAL_MODEL_PATH)


def predict(
    price_model: Pipeline,
    rental_model: Pipeline,
    X: pd.DataFrame,
) -> dict[str, float]:
    price = float(price_model.predict(X)[0])
    rental_yield = float(np.clip(rental_model.predict(X)[0], 0.01, 0.20))
    annual_rental = price * rental_yield
    monthly_rental = annual_rental / 12

    return {
        "predicted_price": round(price, 2),
        "predicted_rental_yield": round(rental_yield, 4),
        "estimated_annual_rental": round(annual_rental, 2),
        "estimated_monthly_rental": round(monthly_rental, 2),
    }


def get_metrics() -> dict[str, Any]:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}
