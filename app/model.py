"""ML model training, prediction, and persistence.

Supports XGBoost, LightGBM, and RandomForest via a VotingRegressor ensemble.
Models are serialised with joblib and metrics written to JSON.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

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

MODEL_VERSION: str = "1.0.0"
CV_N_SPLITS: int = 5
CV_RANDOM_STATE: int = 42
N_STUB_SAMPLES: int = 100
N_STUB_FEATURES: int = 8

if not _HAS_XGB:
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


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple[Any, dict[str, float]]:
    """Train the forecasting model with 5-fold CV; return fitted pipeline and metrics."""
    feature_pipe = build_feature_pipeline()
    X_feat = feature_pipe.fit_transform(X)

    estimator = _build_estimator()
    kf = KFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
    cv_scores = cross_val_score(estimator, X_feat, y, cv=kf, scoring="r2")
    estimator.fit(X_feat, y)

    preds = estimator.predict(X_feat)
    mae = float(np.mean(np.abs(preds - y.values)))
    rmse = float(np.sqrt(np.mean((preds - y.values) ** 2)))

    metrics: dict[str, float] = {
        "model_version": MODEL_VERSION,
        "r2_mean": float(cv_scores.mean()),
        "r2_std": float(cv_scores.std()),
        "mae_kwh": mae,
        "mae_mean": mae,
        "rmse_mean": rmse,
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
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        logger.exception("Failed to load forecasting model from %s", MODEL_PATH)
        return None


def load_anomaly_model() -> dict[str, Any] | None:
    """Load the anomaly detection bundle or return None."""
    if not ANOMALY_MODEL_PATH.exists():
        return None
    try:
        return joblib.load(ANOMALY_MODEL_PATH)
    except Exception:
        logger.exception("Failed to load anomaly model from %s", ANOMALY_MODEL_PATH)
        return None


def predict(arg1: Any, arg2: Any) -> np.ndarray:
    """Run forecasting model on feature DataFrame.

    Accepts arguments in either order: predict(bundle, df) or predict(df, bundle).
    """
    if isinstance(arg1, pd.DataFrame):
        X, model_bundle = arg1, arg2
    else:
        model_bundle, X = arg1, arg2
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


def load_metrics() -> dict[str, Any]:
    """Load latest training metrics, returning a note dict when no file exists.

    Returns:
        Dict of metric name to value, or ``{"note": "no metrics file"}`` when absent.
    """
    if not METRICS_PATH.exists():
        return {"note": "no metrics file"}
    with open(METRICS_PATH) as fh:
        return json.load(fh)


__all__ = [
    "get_feature_importance",
    "get_metrics",
    "load_anomaly_model",
    "load_metrics",
    "load_model",
    "predict",
    "score_anomaly",
    "train_anomaly_model",
    "train_model",
]


def get_feature_importance(model_bundle: dict[str, Any], top_n: int = 20) -> list[dict[str, object]]:
    """Extract feature importances from the trained ensemble, if available.

    Works with RandomForestRegressor and XGBRegressor estimators in a VotingRegressor.
    Falls back to an empty list when the estimator does not expose feature_importances_.

    Args:
        model_bundle: Bundle returned by train_model(), containing 'pipeline' and 'estimator'.
        top_n: Maximum number of features to return (ranked highest to lowest).

    Returns:
        List of dicts with 'feature' and 'importance' keys, sorted descending.
    """
    estimator = model_bundle.get("estimator")
    pipeline = model_bundle.get("pipeline")
    if estimator is None or pipeline is None:
        return []

    importances: np.ndarray | None = None
    try:
        if hasattr(estimator, "estimators_"):
            # VotingRegressor — average importances across sub-estimators
            sub_imps = [e.feature_importances_ for _, e in estimator.estimators_ if hasattr(e, "feature_importances_")]
            if sub_imps:
                importances = np.mean(sub_imps, axis=0)
        elif hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
    except Exception:
        logger.exception("Failed to extract feature importances")
        return []

    if importances is None:
        return []

    try:
        feature_names: list[str] = list(pipeline.get_feature_names_out())
    except Exception:
        feature_names = [f"f{i}" for i in range(len(importances))]

    paired = sorted(
        zip(feature_names, importances.tolist(), strict=False),
        key=lambda x: x[1],
        reverse=True,
    )
    return [{"feature": name, "importance": round(imp, 6)} for name, imp in paired[:top_n]]


def prediction_confidence(
    bundle: dict,
    X: pd.DataFrame,
    n_estimates: int = 10,
) -> dict[str, float]:
    """Estimate prediction confidence via bootstrap sampling of the voting ensemble.

    Runs *n_estimates* random sub-predictions from the ensemble members and
    returns the spread as a confidence proxy.

    Args:
        bundle: Fitted model bundle as returned by :func:`train_model`.
        X: Single-row DataFrame to predict on.
        n_estimates: Number of bootstrap samples to draw.

    Returns:
        Dict with 'mean', 'std', 'lower_95', and 'upper_95' prediction bounds.
    """
    rng = np.random.default_rng(0)
    pipeline = bundle.get("pipeline")
    model = bundle.get("model")
    if pipeline is None or model is None:
        return {"mean": 0.0, "std": 0.0, "lower_95": 0.0, "upper_95": 0.0}

    X_feat = pipeline.transform(X)
    if hasattr(X_feat, "values"):
        X_arr = X_feat.values.astype(float)
    else:
        X_arr = np.array(X_feat, dtype=float)

    estimates = []
    estimators = getattr(model, "estimators_", None)
    if estimators and len(estimators) > 1:
        for _ in range(n_estimates):
            idx = rng.integers(0, len(estimators))
            sub_model = estimators[idx]
            try:
                pred = float(sub_model.predict(X_arr)[0])
                estimates.append(pred)
            except Exception:
                pass
    if not estimates:
        base = float(model.predict(X_arr)[0])
        estimates = [base]

    mean_pred = float(np.mean(estimates))
    std_pred = float(np.std(estimates)) if len(estimates) > 1 else 0.0
    margin = 1.96 * std_pred
    return {
        "mean": round(mean_pred, 4),
        "std": round(std_pred, 4),
        "lower_95": round(max(0.0, mean_pred - margin), 4),
        "upper_95": round(mean_pred + margin, 4),
    }
