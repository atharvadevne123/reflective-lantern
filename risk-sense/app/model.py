"""ML model training, prediction, and serialisation for Risk-Sense."""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "/tmp/risk_sense_model.pkl")
PIPELINE_PATH = os.getenv("PIPELINE_PATH", "/tmp/risk_sense_pipeline.pkl")


def _make_ensemble() -> VotingClassifier:
    """Build XGBoost + LightGBM + RandomForest soft-voting ensemble."""
    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
    except ImportError:
        xgb = None

    try:
        from lightgbm import LGBMClassifier

        lgbm = LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
    except ImportError:
        lgbm = None

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)

    estimators = [("rf", rf)]
    if xgb is not None:
        estimators.append(("xgb", xgb))
    if lgbm is not None:
        estimators.append(("lgbm", lgbm))

    return VotingClassifier(estimators=estimators, voting="soft")


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    pipeline: Any | None = None,
) -> tuple[Any, dict[str, float]]:
    """Train ensemble, run 5-fold CV, return (model, metrics)."""
    ensemble = _make_ensemble()
    calibrated = CalibratedClassifierCV(ensemble, cv=3, method="isotonic")

    X_fit = pipeline.fit_transform(X_train, y_train) if pipeline else X_train

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(calibrated, X_fit, y_train, cv=cv, scoring="roc_auc")

    calibrated.fit(X_fit, y_train)

    train_pred = calibrated.predict_proba(X_fit)[:, 1]
    train_auc = roc_auc_score(y_train, train_pred)

    metrics = {
        "cv_auc_mean": float(cv_scores.mean()),
        "cv_auc_std": float(cv_scores.std()),
        "train_auc": float(train_auc),
        "n_samples": int(len(y_train)),
        "positive_rate": float(y_train.mean()),
    }
    logger.info("Trained model: cv_auc=%.4f±%.4f", metrics["cv_auc_mean"], metrics["cv_auc_std"])
    return calibrated, metrics


def predict(model: Any, X: np.ndarray, pipeline: Any | None = None) -> np.ndarray:
    """Return default probability for each row in X."""
    X_transformed = pipeline.transform(X) if pipeline else X
    proba = model.predict_proba(X_transformed)
    return proba[:, 1]


def save_model(model: Any, pipeline: Any | None = None) -> None:
    """Persist model and optional pipeline to disk."""
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    if pipeline is not None:
        with open(PIPELINE_PATH, "wb") as f:
            pickle.dump(pipeline, f)
    logger.info("Model saved to %s", MODEL_PATH)


def load_model() -> tuple[Any, Any | None]:
    """Load model and optional pipeline from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    pipeline = None
    if os.path.exists(PIPELINE_PATH):
        with open(PIPELINE_PATH, "rb") as f:
            pipeline = pickle.load(f)
    return model, pipeline


def generate_synthetic_data(n_samples: int = 1000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic credit application data for testing."""
    rng = np.random.default_rng(seed)
    n = n_samples
    X = np.column_stack(
        [
            rng.uniform(0.1, 0.8, n),  # debt_to_income
            rng.uniform(0.0, 1.0, n),  # payment_history_score
            rng.uniform(0.0, 0.95, n),  # credit_utilization
            rng.integers(6, 300, n).astype(float),  # account_age_months
            rng.integers(1, 15, n).astype(float),  # num_open_accounts
            rng.integers(0, 5, n).astype(float),  # num_derogatory_marks
            rng.uniform(20_000, 200_000, n),  # annual_income
        ]
    )
    # Higher risk correlated with high DTI, high utilization, low payment score, derogatory marks
    score = (
        0.3 * X[:, 0]  # debt_to_income
        - 0.4 * X[:, 1]  # payment_history_score (inverse)
        + 0.3 * X[:, 2]  # credit_utilization
        + 0.2 * (X[:, 5] / 4)  # derogatory_marks
        - 0.1 * (X[:, 6] / 200_000)  # annual_income (inverse)
    )
    prob = 1 / (1 + np.exp(-5 * (score - 0.3)))
    y = rng.binomial(1, prob, n)
    return X, y
