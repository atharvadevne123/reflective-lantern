"""XGBoost + LightGBM ensemble model for hotel booking demand forecasting."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .features import HotelFeatureEngineer

logger = logging.getLogger(__name__)

MODEL_PATH = Path("model.joblib")
LGBM_PATH = Path("lgbm_model.joblib")
METRICS_PATH = Path("metrics.json")

# Dynamic pricing envelope: multiplier = PRICE_FLOOR + demand * PRICE_SPAN
PRICE_FLOOR: float = 0.70
PRICE_SPAN: float = 0.90

# Demand tier cutoffs on the ensemble probability
HIGH_DEMAND_THRESHOLD: float = 0.70
MEDIUM_DEMAND_THRESHOLD: float = 0.40

# Seasonal demand index per calendar month (for synthetic data generation)
_SEASON_SCORE: dict[int, float] = {
    1: 0.70,
    2: 0.70,
    3: 1.00,
    4: 1.00,
    5: 1.00,
    6: 1.40,
    7: 1.40,
    8: 1.40,
    9: 0.90,
    10: 0.90,
    11: 0.90,
    12: 0.70,
}


def generate_sample_data(n: int = 2000) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate synthetic hotel booking training data.

    Args:
        n: Number of synthetic booking records to generate.

    Returns:
        Tuple of (feature DataFrame, binary target array).
    """
    rng = np.random.default_rng(42)

    months = rng.integers(1, 13, n)
    season_scores = np.array([_SEASON_SCORE[int(m)] for m in months])
    lead_times = rng.integers(0, 181, n)
    room_rates = rng.uniform(80.0, 400.0, n)
    competitor_rates = room_rates * rng.uniform(0.8, 1.25, n)
    occ_rates = (season_scores * rng.uniform(0.5, 0.95, n)).clip(0, 1)

    df = pd.DataFrame(
        {
            "lead_time": lead_times,
            "length_of_stay": rng.integers(1, 8, n),
            "guests_count": rng.integers(1, 5, n),
            "checkin_month": months,
            "checkin_dayofweek": rng.integers(0, 7, n),
            "is_weekend": rng.integers(0, 2, n),
            "current_occ_rate": occ_rates,
            "prev_year_occ_rate": (occ_rates * rng.uniform(0.85, 1.10, n)).clip(0, 1),
            "room_rate": room_rates,
            "competitor_avg_rate": competitor_rates,
            "special_event": rng.integers(0, 2, n),
            "room_type": rng.choice(["standard", "deluxe", "suite"], n),
            "booking_channel": rng.choice(["direct", "online", "ota"], n),
        }
    )

    rate_ratio = (room_rates / competitor_rates.clip(1)).clip(0.5, 2.0)
    booking_prob = (
        season_scores * 0.40
        + (1.0 - rate_ratio) * 0.30
        + (1.0 - lead_times / 180.0) * 0.20
        + occ_rates * 0.10
    ).clip(0.05, 0.95)

    y = (rng.uniform(0, 1, n) < booking_prob).astype(int)
    return df, y


def _make_xgb_pipe() -> Pipeline:
    """Build an XGBoost classifier pipeline with feature engineering and scaling."""
    return Pipeline(
        [
            ("engineer", HotelFeatureEngineer()),
            ("scaler", StandardScaler()),
            (
                "model",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="auc",
                    random_state=42,
                ),
            ),
        ]
    )


def _make_lgbm_pipe() -> Pipeline:
    """Build a LightGBM classifier pipeline with feature engineering and scaling."""
    return Pipeline(
        [
            ("engineer", HotelFeatureEngineer()),
            ("scaler", StandardScaler()),
            (
                "model",
                LGBMClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1,
                ),
            ),
        ]
    )


def train_model(X: pd.DataFrame, y: np.ndarray) -> tuple[Pipeline, Pipeline, dict[str, object]]:
    """Train XGBoost + LightGBM ensemble with 5-fold CV.

    Args:
        X: Raw feature DataFrame (pre-engineering).
        y: Binary booking target array (1 = booked, 0 = not booked).

    Returns:
        Tuple of (xgb_pipeline, lgbm_pipeline, metrics_dict).
    """
    xgb_pipe = _make_xgb_pipe()
    lgbm_pipe = _make_lgbm_pipe()

    logger.info("Running 5-fold CV for XGBoost ...")
    xgb_cv = cross_val_score(xgb_pipe, X, y, cv=5, scoring="roc_auc")
    logger.info("Running 5-fold CV for LightGBM ...")
    lgbm_cv = cross_val_score(lgbm_pipe, X, y, cv=5, scoring="roc_auc")

    xgb_pipe.fit(X, y)
    lgbm_pipe.fit(X, y)

    joblib.dump(xgb_pipe, MODEL_PATH)
    joblib.dump(lgbm_pipe, LGBM_PATH)

    metrics: dict[str, object] = {
        "auc_mean": float((xgb_cv.mean() + lgbm_cv.mean()) / 2),
        "auc_std": float((xgb_cv.std() + lgbm_cv.std()) / 2),
        "xgb_auc_mean": float(xgb_cv.mean()),
        "xgb_auc_std": float(xgb_cv.std()),
        "lgbm_auc_mean": float(lgbm_cv.mean()),
        "lgbm_auc_std": float(lgbm_cv.std()),
        "n_features": 18,
        "n_samples": int(len(y)),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info(
        "Training complete — ensemble AUC %.4f ± %.4f",
        metrics["auc_mean"],
        metrics["auc_std"],
    )
    return xgb_pipe, lgbm_pipe, metrics


def load_models() -> tuple[Pipeline, Pipeline | None, dict[str, object]]:
    """Load persisted models or train fresh ones if absent.

    Returns:
        Tuple of (xgb_pipeline, lgbm_pipeline_or_None, metrics_dict).
    """
    if not MODEL_PATH.exists():
        logger.info("No persisted model found — training on synthetic data")
        X, y = generate_sample_data()
        return train_model(X, y)

    xgb_pipe = joblib.load(MODEL_PATH)
    lgbm_pipe = joblib.load(LGBM_PATH) if LGBM_PATH.exists() else None
    metrics: dict[str, object] = (
        json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    )
    logger.info("Loaded persisted models (version in metrics.json)")
    return xgb_pipe, lgbm_pipe, metrics


def predict_demand(
    xgb_model: Pipeline,
    lgbm_model: Pipeline | None,
    X: pd.DataFrame,
    base_rate: float = 150.0,
) -> dict[str, object]:
    """Ensemble predict demand score and derive optimal room rate.

    Args:
        xgb_model: Fitted XGBoost pipeline.
        lgbm_model: Fitted LightGBM pipeline (may be None).
        X: Raw feature DataFrame for one or more requests.
        base_rate: Baseline room rate used for dynamic pricing calculation.

    Returns:
        Dict containing demand_score, demand_tier, suggested_rate, and
        per_row_scores for drift monitoring.
    """
    xgb_probs: np.ndarray = xgb_model.predict_proba(X)[:, 1]

    if lgbm_model is not None:
        lgbm_probs: np.ndarray = lgbm_model.predict_proba(X)[:, 1]
        per_row = (xgb_probs + lgbm_probs) / 2.0
    else:
        per_row = xgb_probs

    demand_score = float(per_row.mean())

    # Dynamic pricing: demand drives a 0.7–1.6× multiplier on the base rate
    price_multiplier = PRICE_FLOOR + demand_score * PRICE_SPAN
    suggested_rate = round(base_rate * price_multiplier, 2)

    if demand_score >= HIGH_DEMAND_THRESHOLD:
        tier = "high"
    elif demand_score >= MEDIUM_DEMAND_THRESHOLD:
        tier = "medium"
    else:
        tier = "low"

    return {
        "demand_score": round(demand_score, 4),
        "demand_tier": tier,
        "suggested_rate": suggested_rate,
        "per_row_scores": per_row.tolist(),
    }
