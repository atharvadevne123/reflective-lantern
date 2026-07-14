"""Feature engineering pipeline for energy consumption forecasting."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """Create lag features for time-series energy data."""

    def __init__(self, lags: list[int] | None = None) -> None:
        self.lags = lags or [1, 2, 3, 6, 12, 24]

    def fit(self, X: pd.DataFrame, y: Any = None) -> LagFeatureTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "consumption_kwh" in df.columns:
            for lag in self.lags:
                df[f"consumption_lag_{lag}h"] = df["consumption_kwh"].shift(lag).fillna(0)
        return df


class RollingStatsTransformer(BaseEstimator, TransformerMixin):
    """Rolling mean, std, min, max over configurable windows."""

    def __init__(self, windows: list[int] | None = None) -> None:
        self.windows = windows or [3, 6, 12, 24]

    def fit(self, X: pd.DataFrame, y: Any = None) -> RollingStatsTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "consumption_kwh" in df.columns:
            for w in self.windows:
                df[f"rolling_mean_{w}h"] = df["consumption_kwh"].rolling(w, min_periods=1).mean()
                df[f"rolling_std_{w}h"] = df["consumption_kwh"].rolling(w, min_periods=1).std().fillna(0)
                df[f"rolling_max_{w}h"] = df["consumption_kwh"].rolling(w, min_periods=1).max()
        return df


class TemporalFeatureTransformer(BaseEstimator, TransformerMixin):
    """Encode hour-of-day, day-of-week, and seasonal cyclical features."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> TemporalFeatureTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "hour_of_day" in df.columns:
            df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
            df["is_peak_morning"] = ((df["hour_of_day"] >= 7) & (df["hour_of_day"] <= 9)).astype(int)
            df["is_peak_evening"] = ((df["hour_of_day"] >= 17) & (df["hour_of_day"] <= 21)).astype(int)
            df["is_offpeak"] = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] <= 5)).astype(int)
        if "day_of_week" in df.columns:
            df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        return df


class WeatherRatioTransformer(BaseEstimator, TransformerMixin):
    """Compute weather-based energy ratio features."""

    def __init__(self, comfort_temp: float = 20.0) -> None:
        self.comfort_temp = comfort_temp

    def fit(self, X: pd.DataFrame, y: Any = None) -> WeatherRatioTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "temperature_c" in df.columns:
            df["temp_deviation"] = (df["temperature_c"] - self.comfort_temp).abs()
            df["cooling_degree"] = (df["temperature_c"] - self.comfort_temp).clip(lower=0)
            df["heating_degree"] = (self.comfort_temp - df["temperature_c"]).clip(lower=0)
        if "humidity_pct" in df.columns:
            df["discomfort_index"] = df.get("temperature_c", 20) * 0.7 + df["humidity_pct"] * 0.3
        return df


class BuildingEncoderTransformer(BaseEstimator, TransformerMixin):
    """Encode building type as ordinal energy intensity scores."""

    BUILDING_INTENSITY: dict[str, float] = {
        "residential": 1.0,
        "commercial": 2.5,
        "industrial": 5.0,
        "data_center": 8.0,
        "hospital": 4.0,
        "school": 1.8,
        "office": 2.0,
        "retail": 3.0,
    }

    def fit(self, X: pd.DataFrame, y: Any = None) -> BuildingEncoderTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "building_type" in df.columns:
            df["building_intensity"] = df["building_type"].map(self.BUILDING_INTENSITY).fillna(2.0)
            df = df.drop(columns=["building_type"])
        return df


class DropRawColumnsTransformer(BaseEstimator, TransformerMixin):
    """Drop raw columns after feature engineering."""

    COLS_TO_DROP = ["meter_id", "timestamp", "consumption_kwh"]

    def fit(self, X: pd.DataFrame, y: Any = None) -> DropRawColumnsTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        cols = [c for c in self.COLS_TO_DROP if c in df.columns]
        return df.drop(columns=cols)


def build_feature_pipeline() -> Pipeline:
    return Pipeline([
        ("lag", LagFeatureTransformer()),
        ("rolling", RollingStatsTransformer()),
        ("temporal", TemporalFeatureTransformer()),
        ("weather", WeatherRatioTransformer()),
        ("building", BuildingEncoderTransformer()),
        ("drop_raw", DropRawColumnsTransformer()),
        ("scaler", StandardScaler()),
    ])


def prepare_dataframe(data: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(data)
    for col in ["temperature_c", "humidity_pct"]:
        if col not in df.columns:
            df[col] = 20.0 if col == "temperature_c" else 50.0
    if "hour_of_day" not in df.columns:
        df["hour_of_day"] = 0
    if "day_of_week" not in df.columns:
        df["day_of_week"] = 0
    if "is_holiday" not in df.columns:
        df["is_holiday"] = False
    if "building_type" not in df.columns:
        df["building_type"] = "residential"
    df["is_holiday"] = df["is_holiday"].astype(int)
    return df
