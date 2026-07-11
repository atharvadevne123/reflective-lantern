"""Feature engineering pipeline for energy consumption forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract hour-of-day, day-of-week, month, and cyclic encodings."""

    def fit(self, X: pd.DataFrame, y=None) -> TemporalFeatureExtractor:
        """No-op fit (stateless transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add cyclic temporal encodings and binary calendar flags to *X*."""
        df = X.copy()
        if "hour" not in df.columns:
            df["hour"] = 0
        if "day_of_week" not in df.columns:
            df["day_of_week"] = 0
        if "month" not in df.columns:
            df["month"] = 1

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_business_hour"] = ((df["hour"] >= 8) & (df["hour"] <= 18) & (df["day_of_week"] < 5)).astype(int)
        return df


class LagFeatureExtractor(BaseEstimator, TransformerMixin):
    """Add lag features for consumption (1h, 2h, 3h, 6h, 12h, 24h, 168h)."""

    LAG_COLS = [1, 2, 3, 6, 12, 24, 168]

    def fit(self, X: pd.DataFrame, y=None) -> LagFeatureExtractor:
        """No-op fit (stateless transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Append lag columns (lag_1h … lag_168h) for consumption_kwh."""
        df = X.copy()
        base = df.get("consumption_kwh", pd.Series(np.zeros(len(df))))
        for lag in self.LAG_COLS:
            col = f"lag_{lag}h"
            if col not in df.columns:
                df[col] = base.shift(lag).fillna(base.mean() if len(base) > 0 else 0.0)
        return df


class RollingStatsExtractor(BaseEstimator, TransformerMixin):
    """Rolling mean, std, min, max over 3h, 6h, 24h windows."""

    WINDOWS = [3, 6, 24]

    def fit(self, X: pd.DataFrame, y=None) -> RollingStatsExtractor:
        """No-op fit (stateless transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Append rolling mean/std/min/max columns over 3h, 6h, and 24h windows."""
        df = X.copy()
        base = df.get("consumption_kwh", pd.Series(np.zeros(len(df))))
        for w in self.WINDOWS:
            rolled = base.rolling(window=w, min_periods=1)
            df[f"roll_mean_{w}h"] = rolled.mean().fillna(base.mean() if len(base) > 0 else 0.0)
            df[f"roll_std_{w}h"] = rolled.std().fillna(0.0)
            df[f"roll_min_{w}h"] = rolled.min().fillna(base.min() if len(base) > 0 else 0.0)
            df[f"roll_max_{w}h"] = rolled.max().fillna(base.max() if len(base) > 0 else 0.0)
        return df


class WeatherFeatureExtractor(BaseEstimator, TransformerMixin):
    """Derive composite weather features: heat index, cooling degree hours."""

    def fit(self, X: pd.DataFrame, y=None) -> WeatherFeatureExtractor:
        """No-op fit (stateless transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add heat_index, cooling/heating degree-hours, and temp-humidity ratio."""
        df = X.copy()
        temp = df.get("temperature_c", pd.Series(np.full(len(df), 20.0)))
        hum = df.get("humidity_pct", pd.Series(np.full(len(df), 50.0)))

        df["temperature_c"] = temp.fillna(20.0)
        df["humidity_pct"] = hum.fillna(50.0)
        df["heat_index"] = df["temperature_c"] + 0.33 * (df["humidity_pct"] / 100 * 6.105) - 4.0
        df["cooling_deg_hours"] = np.maximum(df["temperature_c"] - 18.0, 0.0)
        df["heating_deg_hours"] = np.maximum(18.0 - df["temperature_c"], 0.0)
        df["temp_humidity_ratio"] = df["temperature_c"] / (df["humidity_pct"] + 1e-6)
        return df


class OccupancyFeatureExtractor(BaseEstimator, TransformerMixin):
    """Encode occupancy and HVAC state into energy-load proxies."""

    def fit(self, X: pd.DataFrame, y=None) -> OccupancyFeatureExtractor:
        """No-op fit (stateless transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add occ_hvac_load proxy and log-occupancy density features."""
        df = X.copy()
        occ = df.get("occupancy", pd.Series(np.zeros(len(df))))
        hvac = df.get("hvac_state", pd.Series(np.zeros(len(df))))
        df["occupancy"] = occ.fillna(0).clip(lower=0)
        df["hvac_state"] = hvac.fillna(0).astype(int)
        df["occ_hvac_load"] = df["occupancy"] * df["hvac_state"]
        df["occupancy_density"] = np.log1p(df["occupancy"])
        return df


class DropNonNumeric(BaseEstimator, TransformerMixin):
    """Drop string/datetime columns before scaling."""

    def fit(self, X: pd.DataFrame, y=None) -> DropNonNumeric:
        """Record which columns are numeric at fit time."""
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Return only the numeric columns as a NumPy array."""
        return X[self.numeric_cols_].values


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """Drops non-numeric or helper columns before model training."""

    DROP_COLS = ["historical_loads", "region", "timestamp"]

    def fit(self, X: pd.DataFrame, y=None) -> DropColumnsTransformer:
        self.cols_to_drop_ = [c for c in self.DROP_COLS if c in X.columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.cols_to_drop_, errors="ignore")


def build_feature_pipeline() -> Pipeline:
    """Build the full sklearn feature engineering pipeline."""
    return Pipeline(
        [
            ("temporal", TemporalFeatureExtractor()),
            ("lag", LagFeatureExtractor()),
            ("rolling", RollingStatsExtractor()),
            ("weather", WeatherFeatureExtractor()),
            ("occupancy", OccupancyFeatureExtractor()),
            ("drop_non_numeric", DropNonNumeric()),
            ("scaler", StandardScaler()),
        ]
    )


def make_feature_row(
    hour: int,
    day_of_week: int,
    month: int,
    temperature_c: float,
    humidity_pct: float,
    occupancy: int,
    hvac_state: int,
    consumption_kwh: float = 0.0,
) -> pd.DataFrame:
    """Build a single-row DataFrame for inference."""
    return pd.DataFrame(
        [
            {
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
                "temperature_c": temperature_c,
                "humidity_pct": humidity_pct,
                "occupancy": occupancy,
                "hvac_state": hvac_state,
                "consumption_kwh": consumption_kwh,
            }
        ]
    )
