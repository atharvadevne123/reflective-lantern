"""ML model training, evaluation, and prediction for Ops-Vision."""

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)

MODEL_PATH: Path = Path(os.environ.get("MODEL_PATH", "/tmp/ops_vision_model.pkl"))
MODEL_VERSION: str = "1.0.0"


def _build_estimators() -> list[tuple[str, Any]]:
    """Return the list of (name, estimator) tuples for the ensemble."""
    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    except ImportError:
        logger.warning("xgboost not available — skipping XGBClassifier")
        xgb = None

    try:
        from lightgbm import LGBMClassifier

        lgbm = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        logger.warning("lightgbm not available — skipping LGBMClassifier")
        lgbm = None

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )

    estimators = [("rf", rf)]
    if xgb is not None:
        estimators.insert(0, ("xgb", xgb))
    if lgbm is not None:
        estimators.append(("lgbm", lgbm))

    return estimators


def build_model() -> VotingClassifier:
    """Construct the soft-voting ensemble classifier.

    Returns:
        Untrained VotingClassifier with XGBoost, LightGBM, and RandomForest.
    """
    estimators = _build_estimators()
    logger.info("Building ensemble with members: %s", [name for name, _ in estimators])
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
) -> tuple[VotingClassifier, dict[str, float]]:
    """Train the ensemble model and return it with 5-fold CV metrics.

    Args:
        X_train: Feature matrix (already processed by the feature pipeline).
        y_train: Binary target labels (1 = incident, 0 = normal).
        cv_folds: Number of stratified cross-validation folds.

    Returns:
        Tuple of (fitted model, metrics dict with mean/std AUC-ROC).
    """
    model = build_model()
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    logger.info("Running %d-fold CV", cv_folds)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
    logger.info("CV AUC-ROC: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    model.fit(X_train, y_train)

    metrics = {
        "cv_auc_mean": float(cv_scores.mean()),
        "cv_auc_std": float(cv_scores.std()),
        "train_samples": int(len(y_train)),
        "positive_rate": float(y_train.mean()),
    }
    return model, metrics


def evaluate(
    model: VotingClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate the model on a held-out test set.

    Args:
        model: Fitted VotingClassifier.
        X_test: Test feature matrix.
        y_test: True labels for test set.

    Returns:
        Dict containing AUC-ROC score on the test set.
    """
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    logger.info("Test AUC-ROC: %.4f", auc)
    return {"test_auc_roc": float(auc)}


def save_model(model: VotingClassifier, path: Path = MODEL_PATH) -> None:
    """Serialise the fitted model to disk.

    Args:
        model: Fitted VotingClassifier to save.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    logger.info("Model saved to %s", path)


def load_model(path: Path = MODEL_PATH) -> VotingClassifier:
    """Load a serialised model from disk.

    Args:
        path: Path to the pickle file.

    Returns:
        Loaded VotingClassifier.

    Raises:
        FileNotFoundError: If no model file exists at path.
    """
    if not path.exists():
        raise FileNotFoundError(f"No model file found at {path}")
    with open(path, "rb") as fh:
        model = pickle.load(fh)
    logger.info("Model loaded from %s", path)
    return model


def predict(
    model: VotingClassifier,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return class predictions and probabilities.

    Args:
        model: Fitted VotingClassifier.
        X: Feature matrix.

    Returns:
        Tuple of (binary predictions, incident probabilities).
    """
    proba = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    return preds, proba


def generate_synthetic_data(
    n_samples: int = 2000,
    incident_rate: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic SRE metric data for development and testing.

    Args:
        n_samples: Number of rows to generate.
        incident_rate: Fraction of samples labelled as incidents.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (features DataFrame, binary label Series).
    """
    rng = np.random.default_rng(random_state)
    n_incident = int(n_samples * incident_rate)
    n_normal = n_samples - n_incident

    normal = {
        "cpu_usage_pct": rng.normal(40, 15, n_normal).clip(0, 100),
        "memory_usage_pct": rng.normal(50, 10, n_normal).clip(0, 100),
        "error_rate_per_min": rng.exponential(2, n_normal).clip(0, 100),
        "latency_p99_ms": rng.normal(150, 50, n_normal).clip(10, 5000),
        "request_rate_per_sec": rng.normal(200, 80, n_normal).clip(1, 2000),
        "disk_io_util_pct": rng.normal(30, 10, n_normal).clip(0, 100),
    }
    incident = {
        "cpu_usage_pct": rng.normal(85, 10, n_incident).clip(0, 100),
        "memory_usage_pct": rng.normal(88, 8, n_incident).clip(0, 100),
        "error_rate_per_min": rng.normal(60, 20, n_incident).clip(0, 200),
        "latency_p99_ms": rng.normal(1500, 500, n_incident).clip(100, 10000),
        "request_rate_per_sec": rng.normal(50, 30, n_incident).clip(1, 2000),
        "disk_io_util_pct": rng.normal(85, 10, n_incident).clip(0, 100),
    }
    df_normal = pd.DataFrame(normal)
    df_incident = pd.DataFrame(incident)
    df = pd.concat([df_normal, df_incident], ignore_index=True)
    labels = pd.Series([0] * n_normal + [1] * n_incident, name="is_incident")
    idx = rng.permutation(len(df))
    return df.iloc[idx].reset_index(drop=True), labels.iloc[idx].reset_index(drop=True)
