"""Ensemble ML model for property valuation.

Trains a soft-voting ensemble of XGBoost, LightGBM, and RandomForest
regressors on the engineered feature array. Runs 5-fold cross-validation
to measure R2 and RMSE before saving the bundle to disk.

Typical usage
-------------
from app.model import train_model, predict, load_model
bundle, metrics = train_model(X_df, y_array)
predictions = predict(X_df, bundle)
"""

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
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from app.features import build_feature_pipeline, extract_feature_array

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "metrics.json"))
MODEL_VERSION = "1.0.0"


def _build_ensemble() -> VotingRegressor:
    """Build the XGBoost + LightGBM + RandomForest soft-voting ensemble.

    Returns:
        An unfitted ``VotingRegressor`` with equal weights across all three
        base learners.
    """
    xgb = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
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
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, n_jobs=-1)
    return VotingRegressor(estimators=[("xgb", xgb), ("lgbm", lgbm), ("rf", rf)])


def train_model(X: pd.DataFrame, y: np.ndarray) -> tuple[Any, dict[str, float]]:
    """Train the ensemble on *X* / *y* and persist the model bundle to disk.

    Runs 5-fold CV to compute R2 and RMSE before fitting on the full dataset.

    Args:
        X: Raw property feature DataFrame (pre-pipeline).
        y: Array of target property values in dollars.

    Returns:
        Tuple of ``(bundle, metrics)`` where *bundle* is a dict containing
        the fitted ``ensemble``, ``scaler``, and ``feature_pipeline``.
    """
    feature_pipeline = build_feature_pipeline()
    feature_pipeline.fit(X)
    X_features = extract_feature_array(X, feature_pipeline)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    ensemble = _build_ensemble()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(ensemble, X_scaled, y, cv=kf, scoring="r2")
    cv_rmse = np.sqrt(
        -cross_val_score(ensemble, X_scaled, y, cv=kf, scoring="neg_mean_squared_error")
    )

    ensemble.fit(X_scaled, y)

    metrics = {
        "r2_mean": float(cv_r2.mean()),
        "r2_std": float(cv_r2.std()),
        "rmse_mean": float(cv_rmse.mean()),
        "rmse_std": float(cv_rmse.std()),
        "n_features": int(X_scaled.shape[1]),
        "n_samples": int(len(y)),
        "model_version": MODEL_VERSION,
    }

    bundle = {"ensemble": ensemble, "scaler": scaler, "feature_pipeline": feature_pipeline}
    joblib.dump(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info(
        "Model trained. R2=%.4f±%.4f, RMSE=%.0f±%.0f",
        cv_r2.mean(),
        cv_r2.std(),
        cv_rmse.mean(),
        cv_rmse.std(),
    )
    return bundle, metrics


def predict(features_df: pd.DataFrame, bundle: dict[str, Any]) -> np.ndarray:
    """Run inference on a batch of property features.

    Args:
        features_df: Raw property feature DataFrame (pre-pipeline).
        bundle: Model bundle returned by :func:`train_model` or :func:`load_model`.

    Returns:
        1-D numpy array of predicted property values.
    """
    X_features = extract_feature_array(features_df, bundle["feature_pipeline"])
    X_scaled = bundle["scaler"].transform(X_features)
    return bundle["ensemble"].predict(X_scaled)


def load_model() -> dict[str, Any]:
    """Load the persisted model bundle from disk, or generate a synthetic stub.

    Returns:
        Dict with ``ensemble``, ``scaler``, and ``feature_pipeline`` keys.
    """
    if not MODEL_PATH.exists():
        logger.warning("No model file found at %s; generating synthetic model", MODEL_PATH)
        return _synthetic_model()
    return joblib.load(MODEL_PATH)


def _synthetic_model() -> dict[str, Any]:
    """Return a minimal stub model for use in test/CI environments."""
    from sklearn.linear_model import Ridge

    X_stub = np.random.default_rng(42).normal(size=(50, 24))
    y_stub = np.random.default_rng(42).uniform(100_000, 1_000_000, size=50)
    scaler = StandardScaler().fit(X_stub)
    ridge = Ridge().fit(scaler.transform(X_stub), y_stub)
    fp = build_feature_pipeline()
    stub_df = pd.DataFrame(
        {
            "sqft": [1500.0],
            "bedrooms": [3],
            "bathrooms": [2.0],
            "lot_size": [5000.0],
            "year_built": [2000],
            "condition_score": [5.0],
            "school_score": [6.0],
            "transit_score": [5.0],
            "walkability_score": [5.0],
            "crime_rate": [0.3],
            "median_neighborhood_price": [300_000.0],
            "median_price_per_sqft": [200.0],
            "avg_rental_yield": [0.06],
            "listing_days": [30],
            "list_price": [350_000.0],
        }
    )
    fp.fit(stub_df)
    bundle = {
        "ensemble": ridge,
        "scaler": scaler,
        "feature_pipeline": fp,
    }
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def load_metrics() -> dict[str, Any]:
    """Return the most recent training metrics from disk.

    Returns:
        Metrics dict, or a minimal placeholder if the file does not exist.
    """
    if not METRICS_PATH.exists():
        return {"model_version": MODEL_VERSION, "note": "no metrics file"}
    return json.loads(METRICS_PATH.read_text())
