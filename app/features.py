"""Feature engineering pipeline for delivery-time and energy prediction."""
from __future__ import annotations

import logging
import math
import statistics
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

CARRIERS = ["DHL", "FedEx", "UPS", "USPS", "Amazon"]
ROUTE_TYPES = ["urban", "suburban", "rural", "highway"]

# Amenity composite weights (sum to 1.0)
_SCHOOL_WEIGHT: float = 0.40
_TRANSIT_WEIGHT: float = 0.35
_WALK_WEIGHT: float = 0.25
_AMENITY_SCALE: float = 10.0


# ---------------------------------------------------------------------------
# Logistics-domain transformers
# ---------------------------------------------------------------------------

class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Adds cyclical and categorical time features.

    Accepts DataFrames with ``hour`` or ``hour_of_day`` (energy/logistics) and
    ``day_of_week``.  All other columns are passed through unchanged.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> TemporalFeatureExtractor:
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        # Support both energy (hour) and logistics (hour_of_day) column names
        if "hour" in df.columns:
            h_col = "hour"
        elif "hour_of_day" in df.columns:
            h_col = "hour_of_day"
        else:
            return df

        hour = df[h_col]
        dow = df.get("day_of_week", pd.Series([0] * len(df), index=df.index))

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        df["is_weekend"] = (dow >= 5).astype(int)
        df["is_business_hour"] = ((hour >= 9) & (hour < 17) & (dow < 5)).astype(int)
        df["is_peak"] = hour.apply(lambda h: 1 if (7 <= h <= 9 or 16 <= h <= 19) else 0)
        logger.debug("Temporal features added: %s", list(df.columns))
        return df


class RouteFeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineers distance buckets, weight ratios, and carrier risk scores."""

    _CARRIER_RISK: dict[str, float] = {
        "DHL": 0.12, "FedEx": 0.10, "UPS": 0.11, "USPS": 0.18, "Amazon": 0.08,
    }

    def fit(self, X: pd.DataFrame, y: Any = None) -> RouteFeatureEngineer:
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if "distance_km" not in X.columns:
            return X.copy()
        df = X.copy()
        df["distance_bucket"] = pd.cut(
            df["distance_km"],
            bins=[0, 30, 150, 800, float("inf")],
            labels=[0, 1, 2, 3],
        ).astype(int)
        df["weight_per_km"] = df["weight_kg"] / (df["distance_km"] + 1e-6)
        if "carrier" in df.columns:
            df["carrier_risk"] = df["carrier"].map(self._CARRIER_RISK).fillna(0.15)
        if "route_type" in df.columns:
            route_map = {r: i for i, r in enumerate(ROUTE_TYPES)}
            df["route_code"] = df["route_type"].map(route_map).fillna(0).astype(int)
        return df


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Ordinal-encodes carrier column."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> CategoricalEncoder:
        if "carrier" not in X.columns:
            self.le_ = None
            return self
        self.le_ = LabelEncoder()
        self.le_.fit(pd.concat([X["carrier"].fillna("Unknown"), pd.Series(CARRIERS)]))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if "carrier" not in X.columns or self.le_ is None:
            return X.copy()
        df = X.copy()
        known = set(self.le_.classes_)
        carriers = df["carrier"].fillna("Unknown").apply(
            lambda c: c if c in known else self.le_.classes_[0]
        )
        df["carrier_enc"] = self.le_.transform(carriers)
        return df


# ---------------------------------------------------------------------------
# Energy-domain transformers
# ---------------------------------------------------------------------------

class LagFeatureExtractor(BaseEstimator, TransformerMixin):
    """Adds lag features for energy consumption (fills missing with median)."""

    LAGS: list[int] = [1, 2, 3, 6, 12, 24, 168]

    def fit(self, X: pd.DataFrame, y: Any = None) -> LagFeatureExtractor:
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        col = "consumption_kwh" if "consumption_kwh" in df.columns else None
        if col is None:
            return df
        series = df[col]
        median_val = float(series.median())
        for lag in self.LAGS:
            shifted = series.shift(lag)
            df[f"lag_{lag}h"] = shifted.fillna(median_val)
        return df


class RollingStatsExtractor(BaseEstimator, TransformerMixin):
    """Adds rolling window statistics for energy consumption."""

    WINDOWS: list[int] = [3, 6, 24]

    def fit(self, X: pd.DataFrame, y: Any = None) -> RollingStatsExtractor:
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        col = "consumption_kwh" if "consumption_kwh" in df.columns else None
        if col is None:
            return df
        series = df[col]
        for w in self.WINDOWS:
            rolled = series.rolling(window=w, min_periods=1)
            df[f"roll_mean_{w}h"] = rolled.mean()
            df[f"roll_std_{w}h"] = rolled.std().fillna(0.0)
            df[f"roll_min_{w}h"] = rolled.min()
            df[f"roll_max_{w}h"] = rolled.max()
        return df


class WeatherFeatureExtractor(BaseEstimator, TransformerMixin):
    """Derives weather-based features from temperature and humidity."""

    BASE_TEMP_COOLING: float = 18.0
    BASE_TEMP_HEATING: float = 15.0

    def fit(self, X: pd.DataFrame, y: Any = None) -> WeatherFeatureExtractor:
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        temp = df.get("temperature_c", pd.Series([20.0] * len(df), index=df.index))
        hum = df.get("humidity_pct", pd.Series([50.0] * len(df), index=df.index))
        df["heat_index"] = temp + 0.33 * (hum / 100 * 6.105 * np.exp(17.27 * temp / (237.3 + temp))) - 4.0
        df["cooling_deg_hours"] = (temp - self.BASE_TEMP_COOLING).clip(lower=0.0)
        df["heating_deg_hours"] = (self.BASE_TEMP_HEATING - temp).clip(lower=0.0)
        df["temp_humidity_ratio"] = temp / (hum + 1e-6)
        return df


class OccupancyFeatureExtractor(BaseEstimator, TransformerMixin):
    """Derives occupancy-based load features."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> OccupancyFeatureExtractor:
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "occupancy" not in df.columns:
            return df
        occ = df["occupancy"]
        hvac = df.get("hvac_state", pd.Series([1] * len(df), index=df.index))
        df["occ_hvac_load"] = occ * hvac
        df["occ_normalized"] = occ / (occ.max() + 1e-6)
        return df


class RatioFeatureTransformer(BaseEstimator, TransformerMixin):
    """Computes real-estate ratio features (beds-per-bath, etc.)."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> RatioFeatureTransformer:
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "bedrooms" in df.columns and "bathrooms" in df.columns:
            df["beds_per_bath"] = df["bedrooms"] / (df["bathrooms"].replace(0, np.nan) + 1e-9)
        if "price" in df.columns and "sqft" in df.columns:
            df["price_per_sqft"] = df["price"] / (df["sqft"] + 1e-6)
        return df


class PropertyAgeTransformer(BaseEstimator, TransformerMixin):
    """Computes property age from year_built."""

    def __init__(self, reference_year: int = 2026) -> None:
        self.reference_year = reference_year

    def fit(self, X: pd.DataFrame, y: Any = None) -> PropertyAgeTransformer:
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "year_built" in df.columns:
            df["property_age"] = self.reference_year - df["year_built"].fillna(self.reference_year)
        return df


class AmenityCompositeTransformer(BaseEstimator, TransformerMixin):
    """Computes a composite amenity score from school, transit, and walk scores."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> AmenityCompositeTransformer:
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        school = df.get("school_score", pd.Series([5.0] * len(df), index=df.index))
        transit = df.get("transit_score", pd.Series([5.0] * len(df), index=df.index))
        walk = df.get("walkability_score", pd.Series([5.0] * len(df), index=df.index))
        df["amenity_composite"] = (
            school * _SCHOOL_WEIGHT + transit * _TRANSIT_WEIGHT + walk * _WALK_WEIGHT
        ) / _AMENITY_SCALE
        if "crime_rate" in df.columns:
            df["amenity_composite"] = df["amenity_composite"] * (1.0 - df["crime_rate"].clip(0.0, 1.0))
        return df


class InteractionFeatureExtractor(BaseEstimator, TransformerMixin):
    """Creates pairwise interaction features for numeric column pairs."""

    PAIRS: list[tuple[str, str]] = [("temperature_c", "occupancy")]

    def fit(self, X: pd.DataFrame, y: Any = None) -> InteractionFeatureExtractor:
        self.available_pairs_ = [
            (a, b) for a, b in self.PAIRS if a in X.columns and b in X.columns
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for a, b in self.available_pairs_:
            df[f"{a}_x_{b}"] = df[a] * df[b]
        return df


# ---------------------------------------------------------------------------
# Pipeline constructors
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "distance_km", "weight_kg", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_peak", "distance_bucket", "weight_per_km", "carrier_risk",
    "route_code", "carrier_enc",
]


def build_feature_pipeline() -> Pipeline:
    """Return a fitted-ready sklearn Pipeline for feature engineering.

    Works for both logistics (carrier, distance_km, hour_of_day) and energy
    (consumption_kwh, temperature_c, occupancy, hour) domain DataFrames.
    """
    return Pipeline([
        ("temporal", TemporalFeatureExtractor()),
        ("weather", WeatherFeatureExtractor()),
        ("occupancy", OccupancyFeatureExtractor()),
        ("lag", LagFeatureExtractor()),
        ("rolling", RollingStatsExtractor()),
        ("amenity", AmenityCompositeTransformer()),
        ("interaction", InteractionFeatureExtractor()),
        ("route", RouteFeatureEngineer()),
        ("categorical", CategoricalEncoder()),
    ])


def prepare_X(df: pd.DataFrame, pipeline: Pipeline, fit: bool = False) -> np.ndarray:
    """Apply the feature pipeline and return the numeric feature matrix."""
    transformed = pipeline.fit_transform(df) if fit else pipeline.transform(df)
    cols = [c for c in FEATURE_COLS if c in transformed.columns]
    return transformed[cols].values.astype(float)


def extract_feature_array(df: pd.DataFrame, pipeline: Pipeline, fit: bool = True) -> np.ndarray:
    """Apply *pipeline* to *df* and return all numeric columns as a 2-D array."""
    transformed = pipeline.fit_transform(df) if fit else pipeline.transform(df)
    if isinstance(transformed, np.ndarray):
        return np.nan_to_num(transformed.astype(float))
    numeric_cols = transformed.select_dtypes(include=[np.number]).columns.tolist()
    return transformed[numeric_cols].fillna(0.0).values.astype(float)


def generate_synthetic_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic delivery records for training and testing."""
    rng = np.random.default_rng(seed)
    carriers = rng.choice(CARRIERS, size=n)
    route_types = rng.choice(ROUTE_TYPES, size=n)
    distance_km = rng.uniform(1, 800, size=n)
    weight_kg = rng.exponential(5, size=n).clip(0.1, 70)
    hour_of_day = rng.integers(0, 24, size=n)
    day_of_week = rng.integers(0, 7, size=n)
    carrier_delay = np.where(np.isin(carriers, ["USPS"]), 1.3, 1.0)
    route_factor = np.where(np.isin(route_types, ["rural"]), 1.4, 1.0)
    target = (distance_km * 1.5 + weight_kg * 2 + rng.normal(0, 20, size=n)) * carrier_delay * route_factor
    target = np.clip(target, 10, 2000)
    return pd.DataFrame({
        "carrier": carriers, "distance_km": distance_km, "weight_kg": weight_kg,
        "route_type": route_types, "hour_of_day": hour_of_day.astype(int),
        "day_of_week": day_of_week.astype(int), "delivery_minutes": target,
    })


# ---------------------------------------------------------------------------
# Numeric utility functions
# ---------------------------------------------------------------------------

def normalize_consumption(values: list[float], method: str = "minmax") -> list[float]:
    """Normalise a list of consumption values.

    Args:
        values: Input values (must not be empty).
        method: ``"minmax"`` scales to [0, 1]; ``"zscore"`` standardises to mean 0.

    Raises:
        ValueError: If *values* is empty or *method* is unrecognised.
    """
    if not values:
        raise ValueError("values must not be empty")
    if method == "minmax":
        mn, mx = min(values), max(values)
        rng = mx - mn
        if rng < 1e-12:
            return [0.0] * len(values)
        return [(v - mn) / rng for v in values]
    if method == "zscore":
        mean = sum(values) / len(values)
        std = statistics.pstdev(values)
        if std < 1e-12:
            return [0.0] * len(values)
        return [(v - mean) / std for v in values]
    raise ValueError(f"method must be 'minmax' or 'zscore', got {method!r}")


def demand_response_potential(
    loads: list[float],
    peak_threshold_pct: float = 0.85,
) -> dict[str, float]:
    """Calculate demand-response potential from hourly load data.

    Args:
        loads: Hourly energy loads in kWh.
        peak_threshold_pct: Fraction of the peak load used as the threshold.

    Returns:
        Dict with ``peak_hours_count``, ``sheddable_kwh``, ``potential_pct``,
        and ``peak_threshold_kwh``.
    """
    if not loads:
        raise ValueError("loads must not be empty")
    if peak_threshold_pct <= 0.0:
        raise ValueError("peak_threshold_pct must be in (0, 1]")
    peak = max(loads)
    threshold = peak * peak_threshold_pct
    peak_hours = [l for l in loads if l >= threshold]
    sheddable = sum(l - threshold for l in peak_hours)
    total = sum(loads)
    potential_pct = sheddable / total if total > 0 else 0.0
    return {
        "peak_hours_count": len(peak_hours),
        "sheddable_kwh": sheddable,
        "potential_pct": min(potential_pct, 1.0),
        "peak_threshold_kwh": threshold,
    }


def encode_cyclical(value: float, max_value: float) -> tuple[float, float]:
    """Return (sin, cos) encoding of *value* on a cycle of *max_value*."""
    angle = 2 * math.pi * value / max_value
    return math.sin(angle), math.cos(angle)


def zscore_feature(values: list[float]) -> list[float]:
    """Return z-score normalised copy of *values*.  Constant series returns all zeros."""
    if not values:
        return []
    mean = sum(values) / len(values)
    std = statistics.pstdev(values)
    if std < 1e-12:
        return [0.0] * len(values)
    return [(v - mean) / std for v in values]


def minmax_normalize(values: list[float]) -> list[float]:
    """Scale *values* to [0, 1].  Constant series returns all zeros."""
    if not values:
        return []
    mn, mx = min(values), max(values)
    rng = mx - mn
    if rng < 1e-12:
        return [0.0] * len(values)
    return [(v - mn) / rng for v in values]


def percentile_feature(
    values: list[float],
    reference: list[float],
    percentile: float = 50.0,
) -> list[float]:
    """Compute percentile rank of each value against a reference distribution.

    Returns 1.0 if *value* >= the *percentile*-th value of *reference*, 0.0 otherwise.
    """
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    if not reference:
        raise ValueError("reference must not be empty")
    sorted_ref = sorted(reference)
    idx = int(len(sorted_ref) * percentile / 100.0)
    idx = min(idx, len(sorted_ref) - 1)
    threshold = sorted_ref[idx]
    return [1.0 if v >= threshold else 0.0 for v in values]


def cumulative_sum_feature(values: list[float]) -> list[float]:
    """Return the cumulative sum of *values*."""
    result: list[float] = []
    running = 0.0
    for v in values:
        running += v
        result.append(running)
    return result


def clip_feature_values(values: list[float], low: float, high: float) -> list[float]:
    """Clip each value in *values* to the range [*low*, *high*].

    Raises:
        ValueError: If *low* > *high*.
    """
    if low > high:
        raise ValueError(f"low ({low}) must be <= high ({high})")
    return [max(low, min(high, v)) for v in values]


def lag_features(values: list[float], lags: list[int]) -> dict[str, list[float | None]]:
    """Compute lag features for a univariate series.

    Args:
        values: Input series.
        lags: List of positive lag integers.

    Returns:
        Dict mapping ``"lag_{k}"`` to a list with *k* leading ``None`` values.

    Raises:
        ValueError: If any lag is not positive.
    """
    for lag in lags:
        if lag < 1:
            raise ValueError(f"All lags must be >= 1, got {lag}")
    result: dict[str, list[float | None]] = {}
    for lag in lags:
        lagged: list[float | None] = [None] * lag + list(values[:-lag] if lag < len(values) else [])
        # Pad to match length
        while len(lagged) < len(values):
            lagged.append(None)
        result[f"lag_{lag}"] = lagged[:len(values)]
    return result


def ratio_feature(numerators: list[float], denominators: list[float]) -> list[float]:
    """Element-wise ratio of two equal-length lists.

    Returns 0.0 when the denominator is zero.

    Raises:
        ValueError: If *numerators* is empty or lengths differ.
    """
    if not numerators:
        raise ValueError("numerators must not be empty")
    if len(numerators) != len(denominators):
        raise ValueError("numerators and denominators must have the same length")
    return [n / d if abs(d) > 1e-12 else 0.0 for n, d in zip(numerators, denominators)]


def bin_feature(values: list[float], bins: list[float]) -> list[int]:
    """Assign each value to a bin index.

    Bin boundaries are right-exclusive: bin 0 is [−∞, bins[0]), bin 1 is
    [bins[0], bins[1]), …, bin n is [bins[n−1], +∞).

    Raises:
        ValueError: If *bins* is not strictly monotonically increasing.
    """
    for i in range(len(bins) - 1):
        if bins[i] >= bins[i + 1]:
            raise ValueError("bins must be strictly increasing")
    result: list[int] = []
    for v in values:
        idx = 0
        for b in bins:
            if v >= b:
                idx += 1
            else:
                break
        result.append(idx)
    return result


def rolling_max_feature(values: list[float], window: int = 3) -> list[float]:
    """Compute rolling maximum with *window* size (min_periods=1).

    Raises:
        ValueError: If *window* is less than 1.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    result: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(max(values[start : i + 1]))
    return result


def difference_feature(values: list[float], order: int = 1) -> list[float]:
    """Return the *order*-th order differences of *values*.

    Raises:
        ValueError: If *order* is less than 1.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    result = list(values)
    for _ in range(order):
        result = [result[i + 1] - result[i] for i in range(len(result) - 1)]
    return result


def rank_features(importances: dict[str, float]) -> list[tuple[str, float]]:
    """Return feature names sorted by importance score descending."""
    return sorted(importances.items(), key=lambda x: x[1], reverse=True)


def top_k_features(importances: dict[str, float], k: int) -> list[str]:
    """Return the names of the top-*k* most important features."""
    ranked = rank_features(importances)
    return [name for name, _ in ranked[:k]]


def make_feature_row(
    hour: int,
    day_of_week: int,
    month: int,
    temperature_c: float,
    humidity_pct: float,
    occupancy: int,
    hvac_state: int,
    consumption_kwh: float,
) -> "np.ndarray":
    """Build a single-row feature array for energy-domain predictions.

    Column order must match the feature columns produced by train_model when
    called with the energy DataFrame schema (all 8 raw features).
    """
    import numpy as np
    return np.array([[hour, day_of_week, month, temperature_c, humidity_pct, occupancy, hvac_state, consumption_kwh]], dtype=float)
