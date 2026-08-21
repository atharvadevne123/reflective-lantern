"""Ensemble ML model for seismic magnitude prediction and aftershock probability."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRegressor

from app.features import build_feature_pipeline, make_synthetic_dataset

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "metrics.json"))

AFTERSHOCK_MAGNITUDE_THRESHOLD = 5.0


def build_ensemble() -> VotingRegressor:
    """Build the weighted XGBoost + RandomForest voting regressor."""
    return VotingRegressor(
        estimators=[
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
                    random_state=42,
                ),
            ),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=150,
                    max_depth=8,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ],
        weights=[0.6, 0.4],
    )


def train_model(
    df: pd.DataFrame | None = None,
    n_samples: int = 2000,
    cv_folds: int = 5,
    persist_metrics: bool = False,
    notes: str | None = None,
) -> tuple[Any, dict[str, float]]:
    """Train the ensemble and report cross-validated and held-out metrics.

    Args:
        df: Training frame with a ``magnitude`` target; synthesised when omitted.
        n_samples: Rows to synthesise when ``df`` is None.
        cv_folds: Cross-validation fold count.
        persist_metrics: Append the run to the ``model_metrics`` table. Off by
            default so unit tests and ad-hoc fits do not touch the database.
        notes: Optional annotation stored alongside a persisted run.

    Returns:
        The fitted pipeline and its metrics dict.
    """
    if df is None:
        df = make_synthetic_dataset(n_samples=n_samples)

    X = df.drop(columns=["magnitude"])
    y = df["magnitude"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    feature_pipeline = build_feature_pipeline()
    ensemble = build_ensemble()

    from sklearn.pipeline import Pipeline

    full_pipeline = Pipeline(steps=[("features", feature_pipeline), ("model", ensemble)])

    cv_scores = cross_val_score(
        full_pipeline, X_train, y_train, cv=cv_folds, scoring="r2", n_jobs=-1
    )

    full_pipeline.fit(X_train, y_train)

    y_pred = full_pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    # Report the post-pipeline width: the raw column count understates what the
    # model actually sees, and it is the engineered width that must stay stable
    # between training and serving.
    n_features = int(full_pipeline.named_steps["features"].transform(X_train.head(1)).shape[1])

    metrics: dict[str, Any] = {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "cv_r2_mean": round(float(cv_scores.mean()), 4),
        "cv_r2_std": round(float(cv_scores.std()), 4),
        "n_features": n_features,
        "n_raw_columns": int(X_train.shape[1]),
        "n_samples": len(df),
        "model_version": "1.0.0",
    }

    joblib.dump(full_pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    logger.info(
        "Model trained — RMSE=%.4f MAE=%.4f R2=%.4f CV-R2=%.4f±%.4f",
        rmse,
        mae,
        r2,
        cv_scores.mean(),
        cv_scores.std(),
    )

    if persist_metrics:
        record_training_run(metrics, notes=notes)

    return full_pipeline, metrics


def record_training_run(metrics: dict[str, Any], notes: str | None = None) -> bool:
    """Append a training run to the ``model_metrics`` table.

    `metrics.json` only ever holds the current champion, so without this the
    history of what was trained and how it scored is lost on every retrain.

    Args:
        metrics: The metrics dict returned by :func:`train_model`.
        notes: Optional free-text annotation for the run.

    Returns:
        ``True`` if the row was written, ``False`` if persistence failed. A
        metrics-logging failure must never abort a successful training run.
    """
    from app.database import ModelMetrics, SessionLocal

    session = SessionLocal()
    try:
        session.add(
            ModelMetrics(
                model_version=str(metrics.get("model_version", "unknown")),
                rmse=float(metrics["rmse"]),
                mae=float(metrics["mae"]),
                r2=float(metrics["r2"]),
                cv_r2_mean=float(metrics["cv_r2_mean"]),
                cv_r2_std=float(metrics["cv_r2_std"]),
                n_features=int(metrics["n_features"]),
                n_samples=int(metrics["n_samples"]),
                notes=notes,
            )
        )
        session.commit()
        return True
    except Exception:
        session.rollback()
        logger.exception("Failed to persist training metrics — model itself is unaffected")
        return False
    finally:
        session.close()


def load_model() -> Any:
    """Load the serialised pipeline, training a fresh one if none exists."""
    if not MODEL_PATH.exists():
        logger.warning("No model found at %s — training from scratch", MODEL_PATH)
        pipeline, _ = train_model()
        return pipeline
    return joblib.load(MODEL_PATH)


def predict_magnitude(pipeline: Any, features: dict[str, Any]) -> dict[str, float]:
    """Score one event, returning magnitude, aftershock probability and class.

    The magnitude is clipped to a physically meaningful 0.1-9.9 range: the
    regressor is unbounded and would otherwise emit values no seismograph
    could produce.
    """
    df = pd.DataFrame([features])
    magnitude = float(pipeline.predict(df)[0])
    magnitude = round(max(0.1, min(9.9, magnitude)), 2)

    # Aftershock probability: logistic function of predicted magnitude
    aftershock_prob = float(1 / (1 + np.exp(-2.5 * (magnitude - AFTERSHOCK_MAGNITUDE_THRESHOLD))))
    aftershock_prob = round(min(0.99, max(0.01, aftershock_prob)), 4)

    return {
        "predicted_magnitude": magnitude,
        "aftershock_probability": aftershock_prob,
        "magnitude_class": classify_magnitude(magnitude),
    }


def classify_magnitude(magnitude: float) -> str:
    """Map a magnitude to its USGS-style descriptive band."""
    if magnitude < 2.0:
        return "micro"
    if magnitude < 4.0:
        return "minor"
    if magnitude < 5.0:
        return "light"
    if magnitude < 6.0:
        return "moderate"
    if magnitude < 7.0:
        return "strong"
    if magnitude < 8.0:
        return "major"
    return "great"


def read_champion_metrics() -> dict[str, float]:
    """Read the incumbent model's metrics, or an empty dict if never trained."""
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text())
