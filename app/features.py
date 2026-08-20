"""Feature engineering pipeline for delivery-time prediction."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

CARRIERS = ["DHL", "FedEx", "UPS", "USPS", "Amazon"]
ROUTE_TYPES = ["urban", "suburban", "rural", "highway"]


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Adds lag, rolling, and cyclical time features."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> TemporalFeatureExtractor:
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        # Cyclical encoding so hour 0 and 23 are adjacent
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        # Weekend flag
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        # Peak-hour flag (7-9 AM and 4-7 PM)
        df["is_peak"] = df["hour_of_day"].apply(
            lambda h: 1 if (7 <= h <= 9 or 16 <= h <= 19) else 0
        )
        logger.debug("Temporal features added: %s", list(df.columns))
        return df


class RouteFeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineers distance buckets, weight ratios, and carrier risk scores."""

    _CARRIER_RISK: dict[str, float] = {
        "DHL": 0.12,
        "FedEx": 0.10,
        "UPS": 0.11,
        "USPS": 0.18,
        "Amazon": 0.08,
    }

    def fit(self, X: pd.DataFrame, y: Any = None) -> RouteFeatureEngineer:
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        # Distance buckets: local, regional, long-haul
        df["distance_bucket"] = pd.cut(
            df["distance_km"],
            bins=[0, 30, 150, 800, float("inf")],
            labels=[0, 1, 2, 3],
        ).astype(int)
        # Weight-to-distance ratio — captures density of effort
        df["weight_per_km"] = df["weight_kg"] / (df["distance_km"] + 1e-6)
        # Carrier delay risk score
        df["carrier_risk"] = df["carrier"].map(self._CARRIER_RISK).fillna(0.15)
        # Route-type encoding
        route_map = {r: i for i, r in enumerate(ROUTE_TYPES)}
        df["route_code"] = df["route_type"].map(route_map).fillna(0).astype(int)
        return df


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Ordinal-encodes carrier column."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> CategoricalEncoder:
        # Trailing underscore matters: sklearn's check_is_fitted only treats
        # attributes ending in "_" as evidence the estimator has been fitted.
        self.le_ = LabelEncoder()
        self.le_.fit(pd.concat([X["carrier"].fillna("Unknown"), pd.Series(CARRIERS)]))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        known = set(self.le_.classes_)
        carriers = df["carrier"].fillna("Unknown").apply(
            lambda c: c if c in known else self.le_.classes_[0]
        )
        df["carrier_enc"] = self.le_.transform(carriers)
        return df


FEATURE_COLS = [
    "distance_km",
    "weight_kg",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_peak",
    "distance_bucket",
    "weight_per_km",
    "carrier_risk",
    "route_code",
    "carrier_enc",
]


def build_feature_pipeline() -> Pipeline:
    """Return a fitted-ready sklearn Pipeline for feature engineering."""
    return Pipeline(
        [
            ("temporal", TemporalFeatureExtractor()),
            ("route", RouteFeatureEngineer()),
            ("categorical", CategoricalEncoder()),
        ]
    )


def prepare_X(df: pd.DataFrame, pipeline: Pipeline, fit: bool = False) -> np.ndarray:
    """Apply the feature pipeline and return the numeric feature matrix."""
    transformed = pipeline.fit_transform(df) if fit else pipeline.transform(df)
    return transformed[FEATURE_COLS].values.astype(float)


def generate_synthetic_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic delivery records for training and testing."""
    rng = np.random.default_rng(seed)
    carriers = rng.choice(CARRIERS, size=n)
    route_types = rng.choice(ROUTE_TYPES, size=n)
    distance_km = rng.uniform(1, 800, size=n)
    weight_kg = rng.exponential(5, size=n).clip(0.1, 70)
    hour_of_day = rng.integers(0, 24, size=n)
    day_of_week = rng.integers(0, 7, size=n)

    # Synthetic target: base time + carrier delay + route penalty + noise
    carrier_delay = np.where(np.isin(carriers, ["USPS"]), 1.3, 1.0)
    route_factor = np.where(np.isin(route_types, ["rural"]), 1.4, 1.0)
    target = (
        distance_km * 1.5
        + weight_kg * 2
        + rng.normal(0, 20, size=n)
    ) * carrier_delay * route_factor
    target = np.clip(target, 10, 2000)

    return pd.DataFrame(
        {
            "carrier": carriers,
            "distance_km": distance_km,
            "weight_kg": weight_kg,
            "route_type": route_types,
            "hour_of_day": hour_of_day.astype(int),
            "day_of_week": day_of_week.astype(int),
            "delivery_minutes": target,
        }
    )
