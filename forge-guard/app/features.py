"""Feature engineering pipeline for manufacturing sensor data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SENSOR_COLS = [
    "temperature",
    "pressure",
    "vibration",
    "cycle_time",
    "tool_wear",
    "power_consumption",
    "humidity",
]

FEATURE_NAMES: list[str] = []


class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """Appends lag-1 and lag-2 features for each sensor column."""

    def __init__(self, lags: int = 2) -> None:
        self.lags = lags

    def fit(self, X: pd.DataFrame, y: object = None) -> LagFeatureTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in SENSOR_COLS:
            if col in df.columns:
                for lag in range(1, self.lags + 1):
                    df[f"{col}_lag{lag}"] = df[col].shift(lag, fill_value=df[col].mean())
        return df


class RollingStatsTransformer(BaseEstimator, TransformerMixin):
    """Adds rolling mean and std over a configurable window."""

    def __init__(self, window: int = 5) -> None:
        self.window = window

    def fit(self, X: pd.DataFrame, y: object = None) -> RollingStatsTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in SENSOR_COLS:
            if col in df.columns:
                df[f"{col}_roll_mean"] = df[col].rolling(self.window, min_periods=1).mean()
                df[f"{col}_roll_std"] = df[col].rolling(self.window, min_periods=1).std().fillna(0)
        return df


class RatioFeatureTransformer(BaseEstimator, TransformerMixin):
    """Computes domain-informed ratio features from sensor readings."""

    def fit(self, X: pd.DataFrame, y: object = None) -> RatioFeatureTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        eps = 1e-9
        if "temperature" in df.columns and "pressure" in df.columns:
            df["temp_pressure_ratio"] = df["temperature"] / (df["pressure"] + eps)
        if "vibration" in df.columns and "cycle_time" in df.columns:
            df["vibration_per_cycle"] = df["vibration"] / (df["cycle_time"] + eps)
        if "tool_wear" in df.columns and "cycle_time" in df.columns:
            df["wear_rate"] = df["tool_wear"] / (df["cycle_time"] + eps)
        if "power_consumption" in df.columns and "cycle_time" in df.columns:
            df["power_efficiency"] = df["power_consumption"] / (df["cycle_time"] + eps)
        return df


class PolynomialSensorTransformer(BaseEstimator, TransformerMixin):
    """Adds squared terms for the highest-signal sensor columns."""

    HIGH_SIGNAL = ["vibration", "tool_wear", "temperature"]

    def fit(self, X: pd.DataFrame, y: object = None) -> PolynomialSensorTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in self.HIGH_SIGNAL:
            if col in df.columns:
                df[f"{col}_sq"] = df[col] ** 2
        return df


class DataFrameToArray(BaseEstimator, TransformerMixin):
    """Converts a DataFrame to a NumPy array and records feature names."""

    def fit(self, X: pd.DataFrame, y: object = None) -> DataFrameToArray:
        global FEATURE_NAMES
        FEATURE_NAMES = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return X.values.astype(np.float64)


def build_feature_pipeline() -> Pipeline:
    """Return an sklearn Pipeline that engineers all features from raw sensor data."""
    return Pipeline(
        [
            ("lag", LagFeatureTransformer(lags=2)),
            ("rolling", RollingStatsTransformer(window=5)),
            ("ratio", RatioFeatureTransformer()),
            ("poly", PolynomialSensorTransformer()),
            ("to_array", DataFrameToArray()),
            ("scaler", StandardScaler()),
        ]
    )


def engineer_single(row: dict[str, float]) -> np.ndarray:
    """Engineer features for a single prediction row (no lag/rolling history)."""
    df = pd.DataFrame([row])
    for col in SENSOR_COLS:
        if col not in df.columns:
            df[col] = 0.0

    eps = 1e-9
    df["temp_pressure_ratio"] = df["temperature"] / (df["pressure"] + eps)
    df["vibration_per_cycle"] = df["vibration"] / (df["cycle_time"] + eps)
    df["wear_rate"] = df["tool_wear"] / (df["cycle_time"] + eps)
    df["power_efficiency"] = df["power_consumption"] / (df["cycle_time"] + eps)

    for col in ["vibration", "tool_wear", "temperature"]:
        df[f"{col}_sq"] = df[col] ** 2

    for col in SENSOR_COLS:
        for lag in range(1, 3):
            df[f"{col}_lag{lag}"] = df[col]
        df[f"{col}_roll_mean"] = df[col]
        df[f"{col}_roll_std"] = 0.0

    return df.values.astype(np.float64)


def generate_synthetic_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic manufacturing sensor data with realistic defect patterns."""
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(
        {
            "temperature": rng.normal(75, 8, n_samples),
            "pressure": rng.normal(50, 5, n_samples),
            "vibration": rng.gamma(2, 1.5, n_samples),
            "cycle_time": rng.normal(30, 3, n_samples),
            "tool_wear": rng.exponential(20, n_samples),
            "power_consumption": rng.normal(100, 10, n_samples),
            "humidity": rng.uniform(30, 70, n_samples),
        }
    )

    defect_score = (
        0.3 * (df["temperature"] > 85).astype(float)
        + 0.25 * (df["vibration"] > 5).astype(float)
        + 0.25 * (df["tool_wear"] > 40).astype(float)
        + 0.15 * (df["pressure"] > 58).astype(float)
        + 0.05 * rng.random(n_samples)
    )
    df["defect"] = (defect_score > 0.4).astype(int)
    return df
