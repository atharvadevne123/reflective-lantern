"""Feature engineering pipeline for energy consumption forecasting."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract hour-of-day, day-of-week, month, and cyclic encodings."""

    def fit(self, X: pd.DataFrame, y: object = None) -> TemporalFeatureExtractor:
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

    def fit(self, X: pd.DataFrame, y: object = None) -> LagFeatureExtractor:
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

    def fit(self, X: pd.DataFrame, y: object = None) -> RollingStatsExtractor:
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

    def fit(self, X: pd.DataFrame, y: object = None) -> WeatherFeatureExtractor:
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

    def fit(self, X: pd.DataFrame, y: object = None) -> OccupancyFeatureExtractor:
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

    def fit(self, X: pd.DataFrame, y: object = None) -> DropNonNumeric:
        """Record which columns are numeric at fit time."""
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return only the numeric columns as a DataFrame."""
        return X[self.numeric_cols_]

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        return np.array(self.numeric_cols_)


class DataFrameWrapper(BaseEstimator, TransformerMixin):
    """Wrap numpy/array output back to a DataFrame, preserving column names."""

    def fit(self, X: pd.DataFrame, y: object = None) -> DataFrameWrapper:
        if hasattr(X, "columns"):
            self.columns_: list[str] = list(X.columns)
        else:
            self.columns_ = [f"f{i}" for i in range(np.array(X).shape[1])]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        arr = np.array(X)
        return pd.DataFrame(arr, columns=self.columns_)


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """Drops non-numeric or helper columns before model training."""

    DROP_COLS = ["historical_loads", "region", "timestamp"]

    def fit(self, X: pd.DataFrame, y: object = None) -> DropColumnsTransformer:
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
            ("to_df", DataFrameWrapper()),
        ]
    )


_SCHOOL_WEIGHT: float = 0.5
_TRANSIT_WEIGHT: float = 0.3
_WALK_WEIGHT: float = 0.2
_AMENITY_SCALE: float = 10.0


class RatioFeatureTransformer(BaseEstimator, TransformerMixin):
    """Compute ratio features such as beds-per-bath."""

    def fit(self, X: pd.DataFrame, y: object = None) -> RatioFeatureTransformer:
        """No fitting required; returns self."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add beds_per_bath column to X."""
        out = X.copy()
        baths = out["bathrooms"] if "bathrooms" in out.columns else pd.Series([1.0] * len(out))
        baths = baths.where(baths != 0, other=1.0)
        beds = out["bedrooms"] if "bedrooms" in out.columns else pd.Series([1.0] * len(out))
        out["beds_per_bath"] = beds / baths
        return out


class PropertyAgeTransformer(BaseEstimator, TransformerMixin):
    """Compute property age from year_built relative to a reference year."""

    def __init__(self, reference_year: int = 2026) -> None:
        self.reference_year = reference_year

    def fit(self, X: pd.DataFrame, y: object = None) -> PropertyAgeTransformer:
        """No fitting required; returns self."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add property_age column to X."""
        out = X.copy()
        out["property_age"] = self.reference_year - out["year_built"]
        return out


class AmenityCompositeTransformer(BaseEstimator, TransformerMixin):
    """Compute a composite amenity score from school, transit, walkability and crime."""

    def fit(self, X: pd.DataFrame, y: object = None) -> AmenityCompositeTransformer:
        """No fitting required; returns self."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add amenity_composite column to X."""
        out = X.copy()
        school = out["school_score"] if "school_score" in out.columns else pd.Series([5.0] * len(out), index=out.index)
        transit = (
            out["transit_score"] if "transit_score" in out.columns else pd.Series([5.0] * len(out), index=out.index)
        )
        walkability = (
            out["walkability_score"]
            if "walkability_score" in out.columns
            else pd.Series([5.0] * len(out), index=out.index)
        )
        out["amenity_composite"] = (
            _SCHOOL_WEIGHT * school + _TRANSIT_WEIGHT * transit + _WALK_WEIGHT * walkability
        ) / _AMENITY_SCALE
        return out


def extract_feature_array(X: pd.DataFrame, pipeline: Pipeline) -> np.ndarray:
    """Apply *pipeline* to *X* and return a 2-D float array.

    Args:
        X: Input DataFrame with raw feature columns.
        pipeline: A fitted or unfitted sklearn Pipeline.

    Returns:
        2-D numpy float array suitable for model input.
    """
    result = pipeline.fit_transform(X)
    if isinstance(result, np.ndarray):
        return result.astype(float)
    return np.array(result, dtype=float)


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


class InteractionFeatureExtractor(BaseEstimator, TransformerMixin):
    """Create pairwise interaction terms between key numeric features."""

    PAIRS: list[tuple[str, str]] = [
        ("temperature_c", "occupancy"),
        ("temperature_c", "hvac_state"),
        ("humidity_pct", "temperature_c"),
        ("hour", "occupancy"),
    ]

    def fit(self, X: pd.DataFrame, y: object = None) -> InteractionFeatureExtractor:
        self.available_pairs_ = [(a, b) for a, b in self.PAIRS if a in X.columns and b in X.columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for a, b in self.available_pairs_:
            df[f"{a}_x_{b}"] = df[a] * df[b]
        return df


def normalize_consumption(
    values: list[float],
    method: str = "minmax",
) -> list[float]:
    """Normalize a consumption series to [0, 1] or zero-mean unit variance.

    Args:
        values: List of consumption readings (kWh).
        method: 'minmax' (default) scales to [0, 1]; 'zscore' standardizes.

    Returns:
        Normalized series of the same length.

    Raises:
        ValueError: If *method* is not 'minmax' or 'zscore'.
        ValueError: If *values* is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    if method not in ("minmax", "zscore"):
        logger.debug("normalize_consumption: invalid method=%r", method)
        raise ValueError(f"method must be 'minmax' or 'zscore', got {method!r}")
    arr = np.array(values, dtype=float)
    if method == "minmax":
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-9:
            return [0.0] * len(values)
        return ((arr - lo) / (hi - lo)).tolist()
    mean, std = arr.mean(), arr.std()
    if std < 1e-9:
        return [0.0] * len(values)
    return ((arr - mean) / std).tolist()


def demand_response_potential(
    hourly_loads: list[float],
    peak_threshold_pct: float = 0.85,
) -> dict[str, object]:
    """Estimate a building's demand-response potential from its hourly load profile.

    Demand response potential is measured as the fraction of hours where load
    exceeds *peak_threshold_pct* x peak load ("peak hours") and the total kWh
    that could be shed if those hours were capped at the threshold.

    Args:
        hourly_loads: Hourly kWh readings (at least one value).
        peak_threshold_pct: Fraction of peak load used as the shedding ceiling
            (default 0.85, i.e. 85 % of peak).

    Returns:
        Dict with:
            - ``peak_hours_count``: Number of hours above threshold.
            - ``sheddable_kwh``: Total kWh that exceeds the threshold.
            - ``potential_pct``: Sheddable kWh as a fraction of total consumption.
            - ``peak_threshold_kwh``: Absolute kWh threshold applied.

    Raises:
        ValueError: If *hourly_loads* is empty or *peak_threshold_pct* is not in (0, 1].
    """
    if not hourly_loads:
        raise ValueError("hourly_loads must not be empty")
    if not (0 < peak_threshold_pct <= 1.0):
        raise ValueError(f"peak_threshold_pct must be in (0, 1], got {peak_threshold_pct}")
    peak = max(hourly_loads)
    threshold = peak * peak_threshold_pct
    peak_hours = [v for v in hourly_loads if v > threshold]
    sheddable = sum(v - threshold for v in peak_hours)
    total = sum(hourly_loads)
    potential_pct = sheddable / total if total > 0 else 0.0
    return {
        "peak_hours_count": len(peak_hours),
        "sheddable_kwh": round(sheddable, 4),
        "potential_pct": round(potential_pct, 4),
        "peak_threshold_kwh": round(threshold, 4),
    }


def encode_cyclical(value: float, max_value: float) -> tuple[float, float]:
    """Encode a cyclical feature as (sin, cos) pair.

    Useful for encoding hour-of-day (max_value=24), day-of-week (7), or month (12)
    without discontinuities at the period boundary.

    Args:
        value: The raw cyclical value.
        max_value: The period length (e.g. 24 for hours).

    Returns:
        (sin_encoded, cos_encoded) tuple.
    """
    sin_val = float(np.sin(2 * np.pi * value / max_value))
    cos_val = float(np.cos(2 * np.pi * value / max_value))
    return round(sin_val, 6), round(cos_val, 6)


def feature_names_for_bundle(bundle: dict) -> list[str]:
    """Return the feature column names expected by the model in *bundle*.

    Extracts column names from the pipeline's feature step if available,
    otherwise returns an empty list.
    """
    model = bundle.get("model")
    if model is None:
        return []
    try:
        return list(model.feature_names_in_)
    except AttributeError:
        pass  # fall through to pipeline alternative
    try:
        return list(model[:-1].get_feature_names_out())
    except Exception:
        return []


__all__ = [
    "AmenityCompositeTransformer",
    "DataFrameWrapper",
    "DropColumnsTransformer",
    "DropNonNumeric",
    "InteractionFeatureExtractor",
    "LagFeatureExtractor",
    "OccupancyFeatureExtractor",
    "PropertyAgeTransformer",
    "RatioFeatureTransformer",
    "RollingStatsExtractor",
    "TemporalFeatureExtractor",
    "WeatherFeatureExtractor",
    "build_feature_pipeline",
    "demand_response_potential",
    "encode_cyclical",
    "extract_feature_array",
    "feature_names_for_bundle",
    "make_feature_row",
    "normalize_consumption",
]


def lag_features(values: list[float], lags: list[int]) -> dict[str, list[float]]:
    """Create lagged feature columns from a time series.

    Args:
        values: Input time series.
        lags: List of lag offsets (positive integers).

    Returns:
        Dict mapping "lag_{n}" -> list of lagged values (NaN as None for missing).

    Raises:
        ValueError: If any lag is non-positive.
    """
    if any(lag <= 0 for lag in lags):
        raise ValueError("All lags must be positive")
    result = {}
    for lag in lags:
        lagged = [None] * lag + list(values[:-lag] if lag < len(values) else [])
        result[f"lag_{lag}"] = lagged[: len(values)]
    return result


def difference_feature(values: list[float], order: int = 1) -> list[float]:
    """Compute nth-order difference of a series.

    Args:
        values: Input numeric series.
        order: Order of differencing. Default 1.

    Returns:
        Differenced series of length max(0, len(values) - order).

    Raises:
        ValueError: If order < 1.
    """
    if order < 1:
        raise ValueError("order must be at least 1")
    result = list(values)
    for _ in range(order):
        result = [result[i] - result[i - 1] for i in range(1, len(result))]
    return result


def ratio_feature(numerator: list[float], denominator: list[float]) -> list[float]:
    """Element-wise ratio of two series; divides numerator by denominator.

    Args:
        numerator: Numerator series.
        denominator: Denominator series.

    Returns:
        List of ratios. Entries where denominator is 0 return 0.0.

    Raises:
        ValueError: If series have different lengths or are empty.
    """
    if not numerator or not denominator:
        raise ValueError("Series must be non-empty")
    if len(numerator) != len(denominator):
        raise ValueError("Series must have the same length")
    return [round(n / d, 6) if d != 0 else 0.0 for n, d in zip(numerator, denominator, strict=False)]


def clip_feature_values(values: list[float], low: float, high: float) -> list[float]:
    """Clip feature values to [low, high] range.

    Args:
        values: Input feature series.
        low: Lower bound.
        high: Upper bound.

    Returns:
        Clipped list of floats.

    Raises:
        ValueError: If low > high.
    """
    if low > high:
        raise ValueError("low must be <= high")
    return [round(max(low, min(high, v)), 6) for v in values]


def bin_feature(values: list[float], bins: list[float]) -> list[int]:
    """Assign each value to a bin index based on sorted bin edges.

    Equivalent to numpy digitize with ``right=False``: bin index 0 means the
    value is below bins[0], index len(bins) means above all edges.

    Args:
        values: Input numeric series.
        bins: Monotonically increasing bin edges.

    Returns:
        List of integer bin indices (0 to len(bins)).

    Raises:
        ValueError: If *bins* is not monotonically increasing.
    """
    if any(bins[i] >= bins[i + 1] for i in range(len(bins) - 1)):
        raise ValueError("bins must be monotonically increasing")
    result: list[int] = []
    for v in values:
        idx = 0
        for edge in bins:
            if v >= edge:
                idx += 1
            else:
                break
        result.append(idx)
    return result


def cumulative_sum_feature(values: list[float]) -> list[float]:
    """Return the cumulative sum of *values* as a feature series.

    Args:
        values: Input numeric series.

    Returns:
        List where each element is the sum of all previous elements including
        the current one.
    """
    result: list[float] = []
    total = 0.0
    for v in values:
        total += v
        result.append(round(total, 6))
    return result


def rolling_max_feature(values: list[float], window: int = 7) -> list[float]:
    """Compute rolling maximum over *window* consecutive values.

    Args:
        values: Input numeric series.
        window: Number of values to include in each rolling window.

    Returns:
        List of rolling maximum values (same length as *values*; earlier
        positions use the available history).

    Raises:
        ValueError: If *window* < 1.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(max(values[start : i + 1]))
    return result


def percentile_feature(
    values: list[float],
    reference: list[float],
    percentile: float = 50.0,
) -> list[float]:
    """Encode each value as its percentile rank within *reference*.

    Args:
        values: Values to encode.
        reference: Reference distribution used to compute the threshold.
        percentile: Target percentile (0-100).

    Returns:
        List of 1.0 (value is at or above the reference percentile) or 0.0.

    Raises:
        ValueError: If *percentile* is outside [0, 100] or *reference* is empty.
    """
    if not reference:
        raise ValueError("reference must not be empty")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    sorted_ref = sorted(reference)
    idx = int(len(sorted_ref) * percentile / 100.0)
    threshold = sorted_ref[min(idx, len(sorted_ref) - 1)]
    return [1.0 if v >= threshold else 0.0 for v in values]


def zscore_feature(values: list[float]) -> list[float]:
    """Convert *values* to per-element z-scores using the series mean and std.

    Args:
        values: Numeric series with at least 2 elements.

    Returns:
        Z-scores of the same length; all zeros when standard deviation is 0.

    Raises:
        ValueError: If *values* has fewer than 2 elements.
    """
    if len(values) < 2:
        raise ValueError(f"values must have at least 2 elements, got {len(values)}")
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5
    if std == 0:
        return [0.0] * len(values)
    return [round((v - mean) / std, 6) for v in values]


def minmax_normalize(values: list[float]) -> list[float]:
    """Min-max normalise *values* to the closed range [0, 1].

    Args:
        values: Numeric series with at least 1 element.

    Returns:
        List of normalised values; ``0.5`` for every position when all
        values are identical (degenerate range).

    Raises:
        ValueError: If *values* is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    lo, hi = min(values), max(values)
    rng = hi - lo
    if rng == 0:
        return [0.5] * len(values)
    return [round((v - lo) / rng, 6) for v in values]
