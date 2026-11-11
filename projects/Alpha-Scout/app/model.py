"""ML model training, prediction, and serialisation."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.features import FEATURE_COLUMNS, StockFeatureEngineer, make_sample_ohlcv

logger = logging.getLogger(__name__)

MODEL_PATH = Path("model.joblib")
METRICS_PATH = Path("metrics.json")
MODEL_VERSION = "1.0.0"


def _build_pipeline() -> Pipeline:
    """Construct the sklearn Pipeline with scaler and XGBClassifier.

    Returns:
        Unfitted sklearn Pipeline.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                n_jobs=1,
            ),
        ),
    ])


def prepare_training_data(n_tickers: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    """Generate labelled training data from synthetic OHLCV."""
    frames = []
    for ticker in ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"][:n_tickers]:
        df = make_sample_ohlcv(300, ticker=ticker)
        eng = StockFeatureEngineer()
        eng.fit(df)
        features = eng.transform(df)
        # Label: 1 if next-period 5-day return > 0, else 0
        future_return = df["close"].pct_change(5).shift(-5)
        labels = (future_return > 0).astype(int)
        valid = features.index.intersection(labels.dropna().index)
        frames.append((features.loc[valid], labels.loc[valid]))

    X = pd.concat([f for f, _ in frames]).reset_index(drop=True)
    y = pd.concat([lbl for _, lbl in frames]).reset_index(drop=True)
    # Keep only columns that exist
    cols = [c for c in FEATURE_COLUMNS if c in X.columns]
    return X[cols].fillna(0), y


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict]:
    pipe = _build_pipeline()
    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc", n_jobs=1)
    pipe.fit(X, y)
    metrics = {
        "auc_mean": float(cv_scores.mean()),
        "auc_std": float(cv_scores.std()),
        "n_features": int(X.shape[1]),
        "n_samples": int(len(y)),
        "version": MODEL_VERSION,
    }
    logger.info("model trained auc_mean=%.4f auc_std=%.4f", metrics["auc_mean"], metrics["auc_std"])
    return pipe, metrics


def save_model(pipe: Pipeline, metrics: dict) -> None:
    joblib.dump(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info("model saved to %s", MODEL_PATH)


def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        logger.warning("model not found — training a new one")
        X, y = prepare_training_data()
        pipe, metrics = train_model(X, y)
        save_model(pipe, metrics)
    return joblib.load(MODEL_PATH)


def predict_alpha(pipe: Pipeline, features: pd.DataFrame) -> dict:
    """Return alpha score, return prediction, and confidence for a feature row."""
    cols = [c for c in FEATURE_COLUMNS if c in features.columns]
    row = features[cols].fillna(0).iloc[[-1]]
    proba = pipe.predict_proba(row)[0]
    positive_class_prob = float(proba[1])
    alpha_score = round(positive_class_prob * 100, 2)
    return {
        "alpha_score": alpha_score,
        "return_prediction": round(positive_class_prob - 0.5, 4),
        "confidence": round(float(np.max(proba)), 4),
        "model_version": MODEL_VERSION,
    }
