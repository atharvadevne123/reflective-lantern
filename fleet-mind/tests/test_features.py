"""Tests for the Fleet-Mind feature engineering pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.features import (
    ALL_FEATURES,
    ENGINEERED_FEATURES,
    RAW_FEATURES,
    FeatureSelector,
    TelemetryFeatureEngineer,
    build_feature_pipeline,
    engineer_single,
)


@pytest.fixture
def base_df():
    """Return a minimal single-row DataFrame with all raw features."""
    return pd.DataFrame([{
        "speed_kmh": 80.0,
        "engine_temp_c": 90.0,
        "oil_pressure_bar": 3.5,
        "fuel_consumption_lph": 8.0,
        "battery_voltage_v": 12.8,
        "brake_pad_mm": 7.0,
        "tire_pressure_bar": 2.3,
        "rpm": 2500.0,
        "coolant_temp_c": 85.0,
        "transmission_temp_c": 75.0,
    }])


class TestTelemetryFeatureEngineer:
    """Tests for TelemetryFeatureEngineer transformer."""

    def test_adds_engineered_columns(self, base_df):
        eng = TelemetryFeatureEngineer()
        result = eng.fit_transform(base_df)
        for col in ENGINEERED_FEATURES:
            assert col in result.columns

    def test_no_raw_column_removed(self, base_df):
        eng = TelemetryFeatureEngineer()
        result = eng.fit_transform(base_df)
        for col in RAW_FEATURES:
            assert col in result.columns

    def test_engine_heat_ratio(self, base_df):
        eng = TelemetryFeatureEngineer()
        result = eng.fit_transform(base_df)
        expected = base_df["engine_temp_c"].values[0] / (base_df["coolant_temp_c"].values[0] + 1e-6)
        assert abs(result["engine_heat_ratio"].values[0] - expected) < 1e-4

    def test_brake_health_score_range(self, base_df):
        eng = TelemetryFeatureEngineer()
        result = eng.fit_transform(base_df)
        assert 0.0 <= result["brake_health_score"].values[0] <= 1.5

    @pytest.mark.parametrize("tire_bar,expected_deviation", [
        (2.4, 0.0),
        (2.0, 0.4),
        (3.0, 0.6),
    ])
    def test_tire_pressure_deviation(self, base_df, tire_bar, expected_deviation):
        df = base_df.copy()
        df["tire_pressure_bar"] = tire_bar
        eng = TelemetryFeatureEngineer()
        result = eng.fit_transform(df)
        assert abs(result["tire_pressure_deviation"].values[0] - expected_deviation) < 1e-6


class TestFeatureSelector:
    """Tests for FeatureSelector transformer."""

    def test_output_shape(self, base_df):
        eng = TelemetryFeatureEngineer()
        enriched = eng.fit_transform(base_df)
        sel = FeatureSelector()
        out = sel.fit_transform(enriched)
        assert out.shape == (1, len(ALL_FEATURES))

    def test_returns_numpy(self, base_df):
        eng = TelemetryFeatureEngineer()
        enriched = eng.fit_transform(base_df)
        sel = FeatureSelector()
        out = sel.fit_transform(enriched)
        assert isinstance(out, np.ndarray)


class TestBuildFeaturePipeline:
    """Tests for build_feature_pipeline()."""

    def test_pipeline_fit_transform(self, base_df):
        pipeline = build_feature_pipeline()
        out = pipeline.fit_transform(base_df)
        assert out.shape == (1, len(ALL_FEATURES))

    def test_pipeline_output_finite(self, base_df):
        pipeline = build_feature_pipeline()
        out = pipeline.fit_transform(base_df)
        assert np.isfinite(out).all()


class TestEngineerSingle:
    """Tests for engineer_single helper."""

    def test_returns_2d_array(self):
        reading = {
            "speed_kmh": 80.0,
            "engine_temp_c": 90.0,
            "oil_pressure_bar": 3.5,
            "fuel_consumption_lph": 8.0,
            "battery_voltage_v": 12.8,
            "brake_pad_mm": 7.0,
            "tire_pressure_bar": 2.3,
            "rpm": 2500.0,
            "coolant_temp_c": 85.0,
            "transmission_temp_c": 75.0,
        }
        out = engineer_single(reading)
        assert out.ndim == 2
        assert out.shape[0] == 1
        assert out.shape[1] == len(ALL_FEATURES)
