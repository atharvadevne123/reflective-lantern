"""Feature engineering pipeline for stock signals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class StockFeatureEngineer(BaseEstimator, TransformerMixin):
    """Derives 20+ technical and fundamental features from raw OHLCV-style input."""

    def fit(self, X: pd.DataFrame, y=None) -> StockFeatureEngineer:
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for lag in [1, 3, 5, 10, 20]:
            df[f"return_lag_{lag}"] = df["close"].pct_change(lag)
        for window in [5, 10, 20]:
            df[f"rolling_mean_{window}"] = df["close"].rolling(window).mean()
            df[f"rolling_std_{window}"] = df["close"].rolling(window).std()
            df[f"rolling_max_{window}"] = df["close"].rolling(window).max()
            df[f"rolling_min_{window}"] = df["close"].rolling(window).min()
        df["price_to_ma5"] = df["close"] / (df["rolling_mean_5"] + 1e-9)
        df["price_to_ma20"] = df["close"] / (df["rolling_mean_20"] + 1e-9)
        df["high_low_ratio"] = df["high"] / (df["low"] + 1e-9)
        df["volume_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)
        df["volatility_5"] = df["return_lag_1"].rolling(5).std()
        df["volatility_20"] = df["return_lag_1"].rolling(20).std()
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        if "pe_ratio" in df.columns:
            df["pe_ratio"] = df["pe_ratio"].fillna(df["pe_ratio"].median())
        else:
            df["pe_ratio"] = 15.0
        if "market_cap" in df.columns:
            df["log_market_cap"] = np.log1p(df["market_cap"])
        else:
            df["log_market_cap"] = 0.0
        raw = {"open", "high", "low", "close", "volume", "ticker", "date"}
        feature_cols = [c for c in df.columns if c not in raw]
        return df[feature_cols].fillna(0)


FEATURE_COLUMNS = [
    "return_lag_1", "return_lag_3", "return_lag_5", "return_lag_10", "return_lag_20",
    "rolling_mean_5", "rolling_mean_10", "rolling_mean_20",
    "rolling_std_5", "rolling_std_10", "rolling_std_20",
    "rolling_max_5", "rolling_max_10", "rolling_max_20",
    "rolling_min_5", "rolling_min_10", "rolling_min_20",
    "price_to_ma5", "price_to_ma20",
    "high_low_ratio", "volume_ratio",
    "volatility_5", "volatility_20",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "pe_ratio", "log_market_cap",
]


def build_feature_pipeline() -> Pipeline:
    return Pipeline([
        ("engineer", StockFeatureEngineer()),
        ("scaler", StandardScaler()),
    ])


def make_sample_ohlcv(n: int = 200, ticker: str = "AAPL") -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.default_rng(42)
    prices = 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
    return pd.DataFrame({
        "ticker": ticker,
        "close": prices,
        "open": prices * (1 + rng.normal(0, 0.003, n)),
        "high": prices * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low": prices * (1 - np.abs(rng.normal(0, 0.005, n))),
        "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
    })
