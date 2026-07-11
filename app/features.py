"""Feature engineering pipeline for energy consumption forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

_SCHOOL_WEIGHT = 0.4
_TRANSIT_WEIGHT = 0.3
_WALK_WEIGHT = 0.3
_AMENITY_SCALE = 10.0

_RENTAL_YIELD_SCALE = 5.0
_AMENITY_IP_WEIGHT = 3.0
_RISK_IP_PENALTY = 2.0

__all__ = [
    "build_feature_pipeline",
    "extract_feature_array",
    "FEATURE_COLUMNS",
    "PropertyAgeTransformer",
    "RatioFeatureTransformer",
    "AmenityCompositeTransformer",
    "InvestmentPotentialTransformer",
    "TierEncoderTransformer",
]

FEATURE_COLUMNS = [
    "sqft",
    "bedrooms",
    "bathrooms",
    "lot_size",
    "year_built",
    "condition_score",
    "school_score",
    "transit_score",
    "walkability_score",
    "crime_rate",
    "median_neighborhood_price",
    "median_price_per_sqft",
    "avg_rental_yield",
    "listing_days",
    "renovation_age",
    "property_age",
    "beds_per_bath",
    "sqft_per_bed",
    "price_ratio_neighborhood",
    "amenity_composite",
    "investment_potential",
    "risk_score",
    "size_tier",
    "age_tier",
]


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract hour-of-day, day-of-week, month, and cyclic encodings."""

    def fit(self, X: pd.DataFrame, y=None) -> TemporalFeatureExtractor:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        school = X.get("school_score", pd.Series(np.ones(len(X)) * 5.0))
        transit = X.get("transit_score", pd.Series(np.ones(len(X)) * 5.0))
        walk = X.get("walkability_score", pd.Series(np.ones(len(X)) * 5.0))
        crime = X.get("crime_rate", pd.Series(np.ones(len(X)) * 0.5))
        X["amenity_composite"] = (school * _SCHOOL_WEIGHT + transit * _TRANSIT_WEIGHT + walk * _WALK_WEIGHT) / _AMENITY_SCALE
        X["risk_score"] = crime.clip(0, 1)
        return X


class WeatherFeatureExtractor(BaseEstimator, TransformerMixin):
    """Derive composite weather features: heat index, cooling degree hours."""

    def fit(self, X: pd.DataFrame, y=None) -> WeatherFeatureExtractor:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        rental_yield = X.get("avg_rental_yield", pd.Series(np.ones(len(X)) * 0.06))
        amenity = X.get("amenity_composite", pd.Series(np.ones(len(X)) * 0.5))
        risk = X.get("risk_score", pd.Series(np.ones(len(X)) * 0.5))
        X["investment_potential"] = (rental_yield * _RENTAL_YIELD_SCALE + amenity * _AMENITY_IP_WEIGHT - risk * _RISK_IP_PENALTY).clip(0, 10)
        return X


class OccupancyFeatureExtractor(BaseEstimator, TransformerMixin):
    """Encode occupancy and HVAC state into energy-load proxies."""

    def fit(self, X: pd.DataFrame, y=None) -> OccupancyFeatureExtractor:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
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
    """Return a fitted-ready sklearn Pipeline of all five feature transformers."""
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
