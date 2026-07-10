"""Feature engineering pipeline for energy load prediction."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

PEAK_HOURS = frozenset(range(7, 22))  # 7am-10pm


class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """Adds historical lag features to the feature dataframe.

    Creates columns for load values at fixed look-back offsets:
    lag_1h, lag_24h, and lag_168h (7 days).

    Args:
        lags: List of integer hour offsets. Defaults to [1, 24, 168].
    """

    def __init__(self, lags: list[int] | None = None) -> None:
        self.lags = lags or [1, 24, 168]

    def fit(self, X: pd.DataFrame, y=None) -> LagFeatureTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "historical_loads" in X.columns:
            history = X["historical_loads"].iloc[0] if hasattr(X["historical_loads"].iloc[0], "__len__") else []
            for lag in self.lags:
                col = f"lag_{lag}h"
                if len(history) >= lag:
                    X[col] = float(history[-lag])
                else:
                    X[col] = 0.0
        else:
            for lag in self.lags:
                X[f"lag_{lag}h"] = 0.0
        return X


class RollingStatsTransformer(BaseEstimator, TransformerMixin):
    """Adds rolling mean and std features over 3h and 24h windows."""

    def __init__(self, windows: list[int] | None = None) -> None:
        self.windows = windows or [3, 24]

    def fit(self, X: pd.DataFrame, y=None) -> RollingStatsTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "historical_loads" in X.columns:
            for w in self.windows:
                col_mean = f"rolling_mean_{w}h"
                col_std = f"rolling_std_{w}h"
                history = X["historical_loads"].iloc[0] if hasattr(X["historical_loads"].iloc[0], "__len__") else []
                if len(history) >= w:
                    window_vals = list(history[-w:])
                    X[col_mean] = float(np.mean(window_vals))
                    X[col_std] = float(np.std(window_vals))
                else:
                    X[col_mean] = 0.0
                    X[col_std] = 0.0
        else:
            for w in self.windows:
                X[f"rolling_mean_{w}h"] = 0.0
                X[f"rolling_std_{w}h"] = 0.0
        return X


class TemporalFeatureTransformer(BaseEstimator, TransformerMixin):
    """Encodes temporal cyclical features: hour, day, month."""

    def fit(self, X: pd.DataFrame, y=None) -> TemporalFeatureTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "hour" in X.columns:
            X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
            X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
        if "day_of_week" in X.columns:
            X["dow_sin"] = np.sin(2 * np.pi * X["day_of_week"] / 7)
            X["dow_cos"] = np.cos(2 * np.pi * X["day_of_week"] / 7)
        if "month" in X.columns:
            X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
            X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
        return X


class PeakPeriodEncoder(BaseEstimator, TransformerMixin):
    """Encodes is_peak_hour and load_category features."""

    def fit(self, X: pd.DataFrame, y=None) -> PeakPeriodEncoder:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "hour" in X.columns:
            X["is_peak_hour"] = X["hour"].apply(lambda h: 1 if h in PEAK_HOURS else 0)
            X["is_morning_peak"] = X["hour"].apply(lambda h: 1 if 7 <= h <= 10 else 0)
            X["is_evening_peak"] = X["hour"].apply(lambda h: 1 if 17 <= h <= 21 else 0)
        if "is_weekend" in X.columns:
            X["is_weekday_peak"] = (
                (X.get("is_peak_hour", 0) == 1) & (X["is_weekend"] == 0)
            ).astype(int)
        return X


class RatioFeatureTransformer(BaseEstimator, TransformerMixin):
    """Computes temperature-humidity interaction ratio and capacity factor."""

    def fit(self, X: pd.DataFrame, y=None) -> RatioFeatureTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "temperature_c" in X.columns and "humidity_pct" in X.columns:
            X["heat_index"] = X["temperature_c"] + 0.33 * (X["humidity_pct"] / 100.0 * 6.105) - 4.0
            X["cooling_demand_proxy"] = np.maximum(X["temperature_c"] - 18.0, 0)
            X["heating_demand_proxy"] = np.maximum(18.0 - X["temperature_c"], 0)
        if "lag_24h" in X.columns and "rolling_mean_24h" in X.columns:
            mean_val = X["rolling_mean_24h"].replace(0, np.nan)
            X["load_ratio_vs_daily_mean"] = X["lag_24h"] / mean_val.fillna(1.0)
        return X


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """Drops non-numeric or helper columns before model training."""

    DROP_COLS = ["historical_loads", "region", "timestamp"]

    def fit(self, X: pd.DataFrame, y=None) -> DropColumnsTransformer:
        self.cols_to_drop_ = [c for c in self.DROP_COLS if c in X.columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.cols_to_drop_, errors="ignore")


def build_feature_pipeline() -> Pipeline:
    return Pipeline([
        ("lag", LagFeatureTransformer()),
        ("rolling", RollingStatsTransformer()),
        ("temporal", TemporalFeatureTransformer()),
        ("peak", PeakPeriodEncoder()),
        ("ratios", RatioFeatureTransformer()),
        ("drop", DropColumnsTransformer()),
        ("scaler", StandardScaler()),
    ])


def build_raw_feature_df(
    hour: int,
    day_of_week: int,
    month: int,
    is_weekend: bool,
    temperature_c: float,
    humidity_pct: float,
    historical_loads: list[float] | None = None,
    region: str = "default",
) -> pd.DataFrame:
    return pd.DataFrame([{
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": int(is_weekend),
        "temperature_c": temperature_c,
        "humidity_pct": humidity_pct,
        "historical_loads": historical_loads or [],
        "region": region,
    }])
