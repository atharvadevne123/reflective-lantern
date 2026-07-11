"""ML model training, prediction, and persistence.

Supports XGBoost, LightGBM, and RandomForest via a VotingRegressor ensemble.
Models are serialised with joblib and metrics written to JSON.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor, VotingRegressor
from sklearn.model_selection import KFold, cross_val_score

try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from app.features import build_feature_pipeline

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model.joblib"))
ANOMALY_MODEL_PATH = Path(os.getenv("ANOMALY_MODEL_PATH", "anomaly_model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "metrics.json"))

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    logger.warning("xgboost not available; using RandomForest only")

def _build_estimator() -> VotingRegressor:
    """Build the voting ensemble from available backends."""
    estimators: list[tuple[str, Any]] = [
        ("rf", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)),
    ]
    if _HAS_XGB:
        estimators.append(
            ("xgb", XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0))
        )
    if _HAS_LGBM:
        estimators.append(
            ("lgbm", LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1))
        )
    return VotingRegressor(estimators=estimators)

CV_N_SPLITS = 5
CV_RANDOM_STATE = 42
N_STUB_SAMPLES = 50
N_STUB_FEATURES = 24


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple[Any, dict[str, float]]:
    """Train the forecasting model with 5-fold CV; return fitted pipeline and metrics."""
    feature_pipe = build_feature_pipeline()
    X_feat = feature_pipe.fit_transform(X)

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
    kf = KFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
    cv_r2 = cross_val_score(ensemble, X_scaled, y, cv=kf, scoring="r2")
    cv_rmse = np.sqrt(
        -cross_val_score(ensemble, X_scaled, y, cv=kf, scoring="neg_mean_squared_error")
    )

    # Compute in-sample MAE
    preds = estimator.predict(X_feat)
    mae = float(np.mean(np.abs(preds - y.values)))

    metrics: dict[str, float] = {
        "r2_mean": float(cv_scores.mean()),
        "r2_std": float(cv_scores.std()),
        "mae_kwh": mae,
        "n_samples": len(y),
        "n_features": X_feat.shape[1],
    }

    joblib.dump({"pipeline": feature_pipe, "estimator": estimator}, MODEL_PATH)
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)

    logger.info("Model trained: R2=%.4f±%.4f  MAE=%.2f kWh", metrics["r2_mean"], metrics["r2_std"], mae)
    return {"pipeline": feature_pipe, "estimator": estimator}, metrics


def train_anomaly_model(X: pd.DataFrame) -> Any:
    """Train an IsolationForest on historical readings for anomaly detection."""
    feature_pipe = build_feature_pipeline()
    X_feat = feature_pipe.fit_transform(X)
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso.fit(X_feat)
    joblib.dump({"pipeline": feature_pipe, "iso": iso}, ANOMALY_MODEL_PATH)
    logger.info("Anomaly model trained on %d samples", len(X))
    return {"pipeline": feature_pipe, "iso": iso}


def load_model() -> dict[str, Any] | None:
    """Load the forecasting model bundle or return None."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def _synthetic_model() -> dict[str, Any]:
    """Return a minimal stub model for use in test/CI environments."""
    from sklearn.linear_model import Ridge

    X_stub = np.random.default_rng(CV_RANDOM_STATE).normal(size=(N_STUB_SAMPLES, N_STUB_FEATURES))
    y_stub = np.random.default_rng(CV_RANDOM_STATE).uniform(100_000, 1_000_000, size=N_STUB_SAMPLES)
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


def predict(model_bundle: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """Run forecasting model on feature DataFrame."""
    feat = model_bundle["pipeline"].transform(X)
    return model_bundle["estimator"].predict(feat)


def score_anomaly(anomaly_bundle: dict[str, Any], X: pd.DataFrame) -> dict[str, Any]:
    """Return anomaly score and flag for a reading."""
    feat = anomaly_bundle["pipeline"].transform(X)
    score = float(anomaly_bundle["iso"].score_samples(feat)[0])
    is_anomaly = int(anomaly_bundle["iso"].predict(feat)[0] == -1)
    severity = "none"
    if is_anomaly:
        severity = "critical" if score < -0.5 else "warning"
    return {"anomaly_score": round(score, 4), "is_anomaly": is_anomaly, "severity": severity}


def get_metrics() -> dict[str, float]:
    """Load latest training metrics."""
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH) as fh:
        return json.load(fh)
