"""Feature engineering pipeline for fleet telemetry data."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

RAW_FEATURES = [
    "speed_kmh",
    "engine_temp_c",
    "oil_pressure_bar",
    "fuel_consumption_lph",
    "battery_voltage_v",
    "brake_pad_mm",
    "tire_pressure_bar",
    "rpm",
    "coolant_temp_c",
    "transmission_temp_c",
]

ENGINEERED_FEATURES = [
    # Ratios
    "engine_heat_ratio",       # engine_temp / coolant_temp
    "oil_efficiency",          # oil_pressure / rpm_scaled
    "fuel_speed_ratio",        # fuel_consumption / (speed + 1)
    "thermal_delta",           # transmission_temp - coolant_temp
    "power_index",             # rpm * speed_kmh / 1e6
    # Normalised brake health
    "brake_health_score",      # brake_pad_mm / 12.0 (new pad = 12mm)
    # Tyre deviation from nominal 2.4 bar
    "tire_pressure_deviation", # abs(tire_pressure_bar - 2.4)
    # Battery health proxy
    "battery_delta",           # abs(battery_voltage_v - 12.6)
]

ALL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES


class TelemetryFeatureEngineer(BaseEstimator, TransformerMixin):
    """Derives engineered features from raw telemetry readings."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> TelemetryFeatureEngineer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to the dataframe."""
        df = X.copy()

        df["engine_heat_ratio"] = df["engine_temp_c"] / (df["coolant_temp_c"] + 1e-6)
        df["oil_efficiency"] = df["oil_pressure_bar"] / ((df["rpm"] / 1000) + 1e-6)
        df["fuel_speed_ratio"] = df["fuel_consumption_lph"] / (df["speed_kmh"] + 1.0)
        df["thermal_delta"] = df["transmission_temp_c"] - df["coolant_temp_c"]
        df["power_index"] = df["rpm"] * df["speed_kmh"] / 1e6
        df["brake_health_score"] = df["brake_pad_mm"] / 12.0
        df["tire_pressure_deviation"] = (df["tire_pressure_bar"] - 2.4).abs()
        df["battery_delta"] = (df["battery_voltage_v"] - 12.6).abs()

        return df


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select only the final feature columns."""

    def __init__(self, feature_names: list[str] | None = None) -> None:
        self.feature_names = feature_names or ALL_FEATURES

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureSelector:
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.feature_names].values


def build_feature_pipeline() -> Pipeline:
    """Return the complete sklearn feature pipeline."""
    return Pipeline(
        steps=[
            ("engineer", TelemetryFeatureEngineer()),
            ("selector", FeatureSelector()),
            ("scaler", StandardScaler()),
        ]
    )


def engineer_single(reading: dict[str, float]) -> np.ndarray:
    """Transform a single telemetry dict into a feature vector."""
    df = pd.DataFrame([reading])
    pipeline = build_feature_pipeline()
    # fit_transform so scaler stats come from the single sample (inference path uses saved scaler)
    return pipeline.fit_transform(df)
