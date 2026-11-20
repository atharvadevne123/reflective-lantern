"""Feature engineering pipeline for market regime detection."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class VolatilityFeature(BaseEstimator, TransformerMixin):
    """Compute rolling volatility (annualised std of log returns)."""

    def __init__(self, window: int = 21) -> None:
        self.window = window

    def fit(self, X: pd.DataFrame, y: Any = None) -> "VolatilityFeature":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        log_ret = np.log(X["close"] / X["close"].shift(1))
        X["volatility"] = log_ret.rolling(self.window).std() * np.sqrt(252)
        return X


class MomentumFeature(BaseEstimator, TransformerMixin):
    """Compute price momentum as % change over a lookback window."""

    def __init__(self, window: int = 21) -> None:
        self.window = window

    def fit(self, X: pd.DataFrame, y: Any = None) -> "MomentumFeature":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["momentum"] = X["close"].pct_change(self.window)
        return X


class VolumeRatioFeature(BaseEstimator, TransformerMixin):
    """Compute volume ratio: current volume vs rolling average."""

    def __init__(self, window: int = 21) -> None:
        self.window = window

    def fit(self, X: pd.DataFrame, y: Any = None) -> "VolumeRatioFeature":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        avg_vol = X["volume"].rolling(self.window).mean()
        X["volume_ratio"] = X["volume"] / (avg_vol + 1e-9)
        return X


class MarketCorrelationFeature(BaseEstimator, TransformerMixin):
    """Compute rolling correlation of asset with a market benchmark."""

    def __init__(self, window: int = 21) -> None:
        self.window = window

    def fit(self, X: pd.DataFrame, y: Any = None) -> "MarketCorrelationFeature":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "market_return" not in X.columns:
            X["correlation"] = 0.0
        else:
            asset_ret = X["close"].pct_change()
            X["correlation"] = asset_ret.rolling(self.window).corr(X["market_return"])
        return X


class BetaFeature(BaseEstimator, TransformerMixin):
    """Compute rolling beta vs market benchmark."""

    def __init__(self, window: int = 63) -> None:
        self.window = window

    def fit(self, X: pd.DataFrame, y: Any = None) -> "BetaFeature":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "market_return" not in X.columns:
            X["beta"] = 1.0
        else:
            asset_ret = X["close"].pct_change()
            mkt = X["market_return"]
            cov = asset_ret.rolling(self.window).cov(mkt)
            var = mkt.rolling(self.window).var()
            X["beta"] = cov / (var + 1e-9)
        return X


class DropRawColumns(BaseEstimator, TransformerMixin):
    """Drop raw OHLCV columns after features are extracted."""

    RAW_COLS = ["open", "high", "low", "close", "volume", "market_return"]

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DropRawColumns":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.RAW_COLS if c in X.columns])


FEATURE_COLS = ["volatility", "momentum", "volume_ratio", "correlation", "beta"]


def build_feature_pipeline() -> Pipeline:
    """Return the full sklearn feature-engineering pipeline."""
    return Pipeline([
        ("volatility", VolatilityFeature()),
        ("momentum", MomentumFeature()),
        ("volume_ratio", VolumeRatioFeature()),
        ("correlation", MarketCorrelationFeature()),
        ("beta", BetaFeature()),
        ("drop_raw", DropRawColumns()),
        ("scaler", StandardScaler()),
    ])


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """Apply feature pipeline and return scaled numeric array."""
    try:
        pipeline = build_feature_pipeline()
        result = pipeline.fit_transform(df)
        logger.info("Features prepared: shape=%s", result.shape)
        return result
    except Exception as e:
        logger.error("Feature preparation failed: %s", e)
        raise


def extract_single_row(row: dict) -> pd.DataFrame:
    """Convert a single prediction request dict into a one-row DataFrame."""
    required = ["close", "volume"]
    for col in required:
        if col not in row:
            raise ValueError(f"Missing required field: {col}")
    defaults = {"open": row["close"], "high": row["close"], "low": row["close"],
                "market_return": 0.0}
    merged = {**defaults, **row}
    df = pd.DataFrame([merged] * 22)  # pad to allow rolling window
    df["close"] = float(merged["close"])
    df["volume"] = float(merged["volume"])
    return df
