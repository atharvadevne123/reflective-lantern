"""Feature engineering pipeline for seismic event data."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

FAULT_TYPES = ["strike_slip", "reverse", "normal", "oblique", "unknown"]

FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "depth_km",
    "station_count",
    "p_wave_amplitude",
    "s_wave_amplitude",
    "epicentral_distance_km",
    "fault_type",
]


class GeoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds geographic and waveform ratio features."""

    def fit(self, X: pd.DataFrame, y=None) -> GeoFeatureEngineer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Distance from major tectonic boundaries (simplified rings)
        df["dist_to_equator"] = np.abs(df["latitude"])
        df["dist_to_dateline"] = np.minimum(
            np.abs(df["longitude"] - 180), np.abs(df["longitude"] + 180)
        )

        # Seismic moment proxy: log(amplitude * distance)
        df["p_seismic_moment"] = np.log1p(
            df["p_wave_amplitude"] * (df["epicentral_distance_km"] + 1)
        )
        df["s_seismic_moment"] = np.log1p(
            df["s_wave_amplitude"] * (df["epicentral_distance_km"] + 1)
        )

        # Wave amplitude ratio (S/P) — correlates with focal mechanism
        df["sp_amplitude_ratio"] = df["s_wave_amplitude"] / (df["p_wave_amplitude"] + 1e-6)

        # Depth-corrected amplitude
        df["depth_corrected_p"] = df["p_wave_amplitude"] / np.log1p(df["depth_km"] + 1)
        df["depth_corrected_s"] = df["s_wave_amplitude"] / np.log1p(df["depth_km"] + 1)

        # Station density proxy
        df["station_density"] = df["station_count"] / (df["epicentral_distance_km"] + 1)

        # Log-transformed depth (shallow quakes behave differently)
        df["log_depth"] = np.log1p(df["depth_km"])

        # Total waveform energy proxy
        df["total_wave_energy"] = np.sqrt(df["p_wave_amplitude"] ** 2 + df["s_wave_amplitude"] ** 2)

        return df


class FaultTypeEncoder(BaseEstimator, TransformerMixin):
    """Ordinal + one-hot encoding for fault type."""

    SEISMICITY_SCORES = {
        "reverse": 0.9,
        "strike_slip": 0.75,
        "oblique": 0.6,
        "normal": 0.5,
        "unknown": 0.4,
    }

    def fit(self, X: pd.DataFrame, y=None) -> FaultTypeEncoder:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        col = df["fault_type"].str.lower().str.strip()

        df["fault_seismicity_score"] = col.map(self.SEISMICITY_SCORES).fillna(0.4)

        for ft in FAULT_TYPES:
            df[f"fault_{ft}"] = (col == ft).astype(int)

        df.drop(columns=["fault_type"], inplace=True, errors="ignore")
        return df


class LagRollingFeatures(BaseEstimator, TransformerMixin):
    """Creates lag and rolling-window features for temporal seismic sequences."""

    def __init__(self, windows: list[int] | None = None) -> None:
        self.windows = windows or [3, 5, 10]

    def fit(self, X: pd.DataFrame, y=None) -> LagRollingFeatures:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        for col in ["p_wave_amplitude", "s_wave_amplitude", "depth_km"]:
            if col not in df.columns:
                continue
            for w in self.windows:
                df[f"{col}_roll_mean_{w}"] = df[col].rolling(window=w, min_periods=1).mean()
                df[f"{col}_roll_std_{w}"] = (
                    df[col].rolling(window=w, min_periods=1).std().fillna(0.0)
                )
            df[f"{col}_lag1"] = df[col].shift(1).fillna(df[col].mean())
            df[f"{col}_lag2"] = df[col].shift(2).fillna(df[col].mean())

        return df


class DropCategoricalColumns(BaseEstimator, TransformerMixin):
    """Drop any remaining non-numeric columns before feeding into the model.

    Selection is by "not numeric" rather than ``dtype == object`` because pandas
    reports string columns as the dedicated ``str`` dtype under the PyArrow-backed
    default, which an object-identity check silently misses.
    """

    def fit(self, X: pd.DataFrame, y=None) -> DropCategoricalColumns:
        self.cols_to_drop_: list[str] = [
            c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.cols_to_drop_, errors="ignore")


class InfinityNaNFixer(BaseEstimator, TransformerMixin):
    """Replace inf/-inf with NaN then fill with column median."""

    def fit(self, X: pd.DataFrame, y=None) -> InfinityNaNFixer:
        tmp = X.replace([np.inf, -np.inf], np.nan)
        self.medians_ = tmp.median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.replace([np.inf, -np.inf], np.nan)
        return df.fillna(self.medians_)


def build_feature_pipeline() -> Pipeline:
    """Assemble the six-stage feature pipeline.

    Stage order matters: the categorical drop and inf/NaN repair must both run
    before ``StandardScaler``, which cannot handle non-numeric or non-finite
    input.
    """
    return Pipeline(
        steps=[
            ("geo_features", GeoFeatureEngineer()),
            ("fault_encoder", FaultTypeEncoder()),
            ("lag_rolling", LagRollingFeatures(windows=[3, 5, 10])),
            ("drop_cat", DropCategoricalColumns()),
            ("fix_nan_inf", InfinityNaNFixer()),
            ("scaler", StandardScaler()),
        ]
    )


def make_synthetic_dataset(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic seismic dataset for training and tests."""
    rng = np.random.default_rng(seed)

    fault_weights = [0.3, 0.25, 0.2, 0.15, 0.1]
    fault_types = rng.choice(FAULT_TYPES, size=n_samples, p=fault_weights)
    fault_scores = np.array([FaultTypeEncoder.SEISMICITY_SCORES[ft] for ft in fault_types])

    depth_km = np.clip(rng.exponential(scale=20, size=n_samples), 0.5, 700)
    station_count = rng.integers(3, 50, size=n_samples)
    epicentral_distance_km = rng.exponential(scale=150, size=n_samples) + 5

    p_wave_amplitude = rng.lognormal(mean=1.0, sigma=1.2, size=n_samples)
    s_wave_amplitude = p_wave_amplitude * rng.uniform(1.1, 2.5, size=n_samples)

    # Ground-truth magnitude approximation (Richter-like relationship)
    magnitude = (
        0.8 * np.log10(p_wave_amplitude + 1)
        + 0.6 * np.log10(s_wave_amplitude + 1)
        - 0.3 * np.log1p(depth_km / 100)
        + 0.5 * fault_scores
        + 0.2 * np.log1p(station_count)
        + rng.normal(0, 0.25, size=n_samples)
    )
    magnitude = np.clip(magnitude, 0.1, 9.5)

    df = pd.DataFrame(
        {
            "latitude": rng.uniform(-60, 60, size=n_samples),
            "longitude": rng.uniform(-180, 180, size=n_samples),
            "depth_km": depth_km,
            "station_count": station_count,
            "p_wave_amplitude": p_wave_amplitude,
            "s_wave_amplitude": s_wave_amplitude,
            "epicentral_distance_km": epicentral_distance_km,
            "fault_type": fault_types,
            "magnitude": magnitude,
        }
    )
    return df
