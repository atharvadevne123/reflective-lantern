"""Feature engineering pipeline for equity trading signals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TechnicalIndicators(BaseEstimator, TransformerMixin):
    """Compute RSI, MACD, Bollinger Bands, ATR, and volume ratio features."""

    def __init__(self, rsi_period: int = 14, bb_period: int = 20) -> None:
        self.rsi_period = rsi_period
        self.bb_period = bb_period

    def fit(self, X: pd.DataFrame, y=None) -> TechnicalIndicators:
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        df = X.copy()
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(self.rsi_period).mean()
        avg_loss = loss.rolling(self.rsi_period).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df["rsi"] = 100 - (100 / (1 + rs))

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        sma = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std()
        df["bb_upper"] = sma + 2 * std
        df["bb_lower"] = sma - 2 * std
        df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (sma + 1e-9)

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

        df["vol_ratio"] = volume / (volume.rolling(20).mean() + 1e-9)

        return df[[
            "rsi", "macd", "macd_signal", "macd_hist",
            "bb_pct", "bb_width", "atr", "vol_ratio",
        ]].fillna(0).values


class LagRollingFeatures(BaseEstimator, TransformerMixin):
    """Add lag returns and rolling stats over price series."""

    def fit(self, X: pd.DataFrame, y=None) -> LagRollingFeatures:
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        df = X.copy()
        close = df["close"].astype(float)

        for lag in [1, 2, 3, 5, 10]:
            df[f"ret_{lag}d"] = close.pct_change(lag)

        df["vol_5d"] = df["ret_1d"].rolling(5).std()
        df["vol_20d"] = df["ret_1d"].rolling(20).std()
        df["sma_ratio_5_20"] = close.rolling(5).mean() / (close.rolling(20).mean() + 1e-9)
        df["sma_ratio_10_50"] = close.rolling(10).mean() / (close.rolling(50).mean() + 1e-9)
        df["momentum_10"] = close / (close.shift(10) + 1e-9) - 1
        df["momentum_20"] = close / (close.shift(20) + 1e-9) - 1

        lag_cols = [
            "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d",
            "vol_5d", "vol_20d", "sma_ratio_5_20", "sma_ratio_10_50",
            "momentum_10", "momentum_20",
        ]
        return df[lag_cols].fillna(0).values


class RatioFeatures(BaseEstimator, TransformerMixin):
    """Compute price-volume ratio and open-close spread features."""

    def fit(self, X: pd.DataFrame, y=None) -> RatioFeatures:
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        df = X.copy()
        close = df["close"].astype(float)
        open_ = df["open"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        df["oc_spread"] = (close - open_) / (open_ + 1e-9)
        df["hl_range"] = (high - low) / (close + 1e-9)
        df["price_position"] = (close - low) / (high - low + 1e-9)

        ratio_cols = ["oc_spread", "hl_range", "price_position"]
        return df[ratio_cols].fillna(0).values


def build_feature_pipeline() -> Pipeline:
    """Return a three-stage feature pipeline producing a numeric array."""
    return Pipeline([
        ("scaler", StandardScaler()),
    ])


def engineer_features(df: pd.DataFrame) -> np.ndarray:
    """Run all feature transformers and concatenate output."""
    tech = TechnicalIndicators().transform(df)
    lag = LagRollingFeatures().transform(df)
    ratio = RatioFeatures().transform(df)
    return np.hstack([tech, lag, ratio])


FEATURE_NAMES = [
    "rsi", "macd", "macd_signal", "macd_hist", "bb_pct", "bb_width", "atr", "vol_ratio",
    "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d",
    "vol_5d", "vol_20d", "sma_ratio_5_20", "sma_ratio_10_50", "momentum_10", "momentum_20",
    "oc_spread", "hl_range", "price_position",
]
