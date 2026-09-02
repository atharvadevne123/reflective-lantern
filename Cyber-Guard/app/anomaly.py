"""Unsupervised anomaly detection for connections the classifier has not seen.

The supervised ensemble in :mod:`app.model` can only assign one of the five
known threat classes. An IsolationForest fitted on the same engineered
features flags connections that sit far outside the training distribution --
the zero-day case, where the right answer is "none of the above".
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from app.features import build_feature_pipeline

logger = logging.getLogger(__name__)

ANOMALY_MODEL_PATH = os.getenv("ANOMALY_MODEL_PATH", "anomaly_model.joblib")

# Fraction of training data assumed anomalous; drives the decision threshold.
DEFAULT_CONTAMINATION = 0.05


def train_anomaly_detector(
    X: pd.DataFrame,
    contamination: float = DEFAULT_CONTAMINATION,
    model_path: str = ANOMALY_MODEL_PATH,
) -> Pipeline:
    """Fit an IsolationForest over the engineered feature space.

    Args:
        X: Raw connection records with the six standard packet columns.
        contamination: Expected proportion of outliers in ``X``.
        model_path: Destination for the serialised pipeline.

    Returns:
        The fitted feature-engineering + IsolationForest pipeline.
    """
    pipe = Pipeline(
        [
            ("features", build_feature_pipeline()),
            (
                "detector",
                IsolationForest(
                    n_estimators=100,
                    contamination=contamination,
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(X)
    joblib.dump(pipe, model_path)
    logger.info("anomaly detector trained n=%d contamination=%.3f", len(X), contamination)
    return pipe


def load_anomaly_detector(model_path: str = ANOMALY_MODEL_PATH) -> Pipeline:
    """Load a serialised anomaly-detection pipeline.

    Args:
        model_path: Path written by :func:`train_anomaly_detector`.

    Returns:
        The deserialised pipeline.
    """
    return joblib.load(model_path)


def score_anomaly(X: pd.DataFrame, pipeline: Pipeline) -> dict[str, Any]:
    """Score a single connection for outlier-ness.

    Args:
        X: A one-row DataFrame of raw connection fields.
        pipeline: A fitted pipeline from :func:`train_anomaly_detector`.

    Returns:
        A dict with ``anomaly_score`` (higher is more anomalous),
        ``is_anomaly``, and the raw signed ``decision_score``.
    """
    decision = float(pipeline.decision_function(X)[0])
    is_anomaly = bool(pipeline.predict(X)[0] == -1)
    return {
        # decision_function is positive for inliers; flip so higher == weirder.
        "anomaly_score": round(-decision, 4),
        "decision_score": round(decision, 4),
        "is_anomaly": is_anomaly,
    }


def ensure_anomaly_model_exists(model_path: str = ANOMALY_MODEL_PATH) -> None:
    """Train and persist an anomaly detector if none exists yet.

    Args:
        model_path: Path checked for an existing model.
    """
    if not Path(model_path).exists():
        from app.model import generate_synthetic_data

        X, _ = generate_synthetic_data(500)
        train_anomaly_detector(X, model_path=model_path)


def batch_anomaly_rate(X: pd.DataFrame, pipeline: Pipeline) -> float:
    """Return the fraction of rows in ``X`` flagged as anomalous.

    A sustained rise in this rate is an early signal of a novel attack
    pattern, ahead of any label being available to retrain on.

    Args:
        X: Raw connection records.
        pipeline: A fitted anomaly pipeline.

    Returns:
        Anomaly rate in ``[0, 1]``; ``0.0`` for an empty frame.
    """
    if len(X) == 0:
        return 0.0
    preds = pipeline.predict(X)
    return float(np.mean(preds == -1))
