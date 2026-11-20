"""ML model training, evaluation, and inference for Property-Sage.

Trains an XGBoost + LightGBM + RandomForest ensemble for both property
price and rental yield prediction using 5-fold cross-validation.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from app.experiment_tracking import log_run
from app.features import PropertyFeatureEngineer, generate_synthetic_data

logger = logging.getLogger(__name__)

MODEL_DIR: Path = Path(os.getenv("MODEL_DIR", "models"))
PRICE_MODEL_PATH: Path = MODEL_DIR / "price_model.joblib"
RENTAL_MODEL_PATH: Path = MODEL_DIR / "rental_model.joblib"
METRICS_PATH: Path = MODEL_DIR / "metrics.json"


def _build_ensemble() -> VotingRegressor:
    """Construct the weighted XGBoost + LightGBM + RandomForest ensemble.

    Returns:
        VotingRegressor with weights [0.4, 0.4, 0.2].
    """
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
    """Return a full sklearn pipeline: feature engineering → scaling → ensemble.

    Returns:
        sklearn Pipeline with three steps: features, scaler, model.
    """
    return Pipeline(
        [
            ("features", PropertyFeatureEngineer()),
            ("scaler", StandardScaler()),
            ("model", _build_ensemble()),
        ]
    )


def train_model(
    X: pd.DataFrame,
    y_price: pd.Series,
    y_rental: pd.Series,
) -> dict[str, Any]:
    """Train price and rental yield models and persist them to disk.

    Runs 5-fold CV to compute R², RMSE, and MAE before saving.

    Args:
        X: Feature DataFrame (raw property attributes).
        y_price: Target sale price Series.
        y_rental: Target rental yield Series (0–1 range).

    Returns:
        Dict of performance metrics for both models.
    """
    MODEL_DIR.mkdir(exist_ok=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    logger.info("Training price model — %d samples, 5-fold CV", len(X))
    price_pipe = _build_pipeline()
    price_cv = cross_val_score(price_pipe, X, y_price, cv=kf, scoring="r2", n_jobs=-1)
    price_pipe.fit(X, y_price)
    price_pred = price_pipe.predict(X)
    price_rmse = float(np.sqrt(mean_squared_error(y_price, price_pred)))
    price_mae = float(mean_absolute_error(y_price, price_pred))

    logger.info("Training rental yield model — %d samples, 5-fold CV", len(X))
    rental_pipe = _build_pipeline()
    rental_cv = cross_val_score(rental_pipe, X, y_rental, cv=kf, scoring="r2", n_jobs=-1)
    rental_pipe.fit(X, y_rental)
    rental_pred = rental_pipe.predict(X)
    rental_rmse = float(np.sqrt(mean_squared_error(y_rental, rental_pred)))
    rental_mae = float(mean_absolute_error(y_rental, rental_pred))

    joblib.dump(price_pipe, PRICE_MODEL_PATH)
    joblib.dump(rental_pipe, RENTAL_MODEL_PATH)

    metrics: dict[str, Any] = {
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
    logger.info(
        "Models saved — price R²=%.4f rental R²=%.4f",
        price_cv.mean(),
        rental_cv.mean(),
    )

    log_run(
        experiment_name="property_price",
        params={"n_estimators": 200, "ensemble": "xgb+lgbm+rf", "n_train": len(X)},
        metrics={k: v for k, v in metrics.items() if "price" in k},
    )
    log_run(
        experiment_name="rental_yield",
        params={"n_estimators": 200, "ensemble": "xgb+lgbm+rf", "n_train": len(X)},
        metrics={k: v for k, v in metrics.items() if "rental" in k},
    )
    return metrics


def load_models() -> tuple[Pipeline, Pipeline]:
    """Load serialised models from disk, training from scratch if absent.

    Returns:
        Tuple of (price_pipeline, rental_yield_pipeline).
    """
    if not PRICE_MODEL_PATH.exists() or not RENTAL_MODEL_PATH.exists():
        logger.info("Serialised models not found — training on synthetic data")
        X, y_price, y_rental = generate_synthetic_data(n=2000)
        train_model(X, y_price, y_rental)
    return joblib.load(PRICE_MODEL_PATH), joblib.load(RENTAL_MODEL_PATH)


def predict(
    price_model: Pipeline,
    rental_model: Pipeline,
    X: pd.DataFrame,
) -> dict[str, float]:
    """Run inference for a single property row.

    Args:
        price_model: Fitted price prediction pipeline.
        rental_model: Fitted rental yield prediction pipeline.
        X: One-row property DataFrame.

    Returns:
        Dict with predicted_price, predicted_rental_yield, and derived rental estimates.
    """
    price = float(price_model.predict(X)[0])
    rental_yield = float(np.clip(rental_model.predict(X)[0], 0.01, 0.20))
    annual_rental = price * rental_yield
    monthly_rental = annual_rental / 12

    logger.debug(
        "Inference complete — price=%.2f yield=%.4f",
        price,
        rental_yield,
    )
    return {
        "predicted_price": round(price, 2),
        "predicted_rental_yield": round(rental_yield, 4),
        "estimated_annual_rental": round(annual_rental, 2),
        "estimated_monthly_rental": round(monthly_rental, 2),
    }


def get_metrics() -> dict[str, Any]:
    """Return the most recently persisted training metrics.

    Returns:
        Dict of metric key-value pairs, or empty dict if no metrics exist.
    """
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


def calculate_confidence_score(
    cv_scores: list[float],
    prediction_value: float,
    reference_mean: float,
    reference_std: float,
) -> float:
    """Estimate a confidence score (0–1) for a prediction.

    Uses CV R² mean and how many stddevs the prediction is from the
    training-data mean. Higher score = higher confidence.

    Args:
        cv_scores: Cross-validation R² scores from training.
        prediction_value: The raw predicted value.
        reference_mean: Mean of the training target distribution.
        reference_std: Std deviation of the training target distribution.

    Returns:
        Confidence score in [0.0, 1.0].
    """
    if not cv_scores:
        return 0.0
    r2_confidence = float(max(0.0, min(1.0, sum(cv_scores) / len(cv_scores))))
    if reference_std <= 0:
        deviation_penalty = 0.0
    else:
        z = abs(prediction_value - reference_mean) / reference_std
        deviation_penalty = float(min(0.5, z / 10.0))
    return round(max(0.0, r2_confidence - deviation_penalty), 4)


def batch_predict(
    price_model: Pipeline,
    rental_model: Pipeline,
    X: pd.DataFrame,
) -> list[dict[str, float]]:
    """Run inference for multiple property rows at once.

    Args:
        price_model: Fitted price prediction pipeline.
        rental_model: Fitted rental yield prediction pipeline.
        X: Multi-row property DataFrame.

    Returns:
        List of prediction dicts, one per row.
    """
    prices = price_model.predict(X).tolist()
    yields = np.clip(rental_model.predict(X), 0.01, 0.20).tolist()
    results = []
    for price, rental_yield in zip(prices, yields):
        price = float(price)
        rental_yield = float(rental_yield)
        annual_rental = price * rental_yield
        results.append({
            "predicted_price": round(price, 2),
            "predicted_rental_yield": round(rental_yield, 4),
            "estimated_annual_rental": round(annual_rental, 2),
            "estimated_monthly_rental": round(annual_rental / 12, 2),
        })
    return results


def get_model_version() -> str:
    """Return the current model version string from METRICS_PATH.

    Returns:
        ISO-format date of the most recent training run, or 'unknown'.
    """
    import datetime
    if METRICS_PATH.exists():
        try:
            mtime = METRICS_PATH.stat().st_mtime
            dt = datetime.datetime.fromtimestamp(mtime)
            return dt.strftime("%Y%m%d_%H%M%S")
        except Exception:
            pass
    return "unknown"


def clamp_prediction(
    value: float,
    lo: float = 10_000.0,
    hi: float = 50_000_000.0,
) -> float:
    """Clamp a price prediction to a plausible real-estate range.

    Args:
        value: Raw predicted value.
        lo: Minimum plausible price.
        hi: Maximum plausible price.

    Returns:
        Value clamped to [lo, hi].
    """
    return max(lo, min(hi, value))
