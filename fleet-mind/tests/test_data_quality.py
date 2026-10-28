"""Tests for fleet telemetry data quality checks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from app.data_quality import assess_batch_quality, check_missing_values, check_outliers_zscore


@pytest.fixture
def clean_readings():
    return [
        {
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
        for _ in range(10)
    ]


class TestCheckMissingValues:
    def test_clean_df_no_missing(self, clean_readings):
        df = pd.DataFrame([{k: v for k, v in r.items()} for r in clean_readings])
        missing = check_missing_values(df)
        assert all(v == 0 for v in missing.values())

    def test_detects_nan(self, clean_readings):
        readings = [{**r, "engine_temp_c": None} for r in clean_readings]
        df = pd.DataFrame(readings)
        missing = check_missing_values(df)
        assert missing.get("engine_temp_c", 0) == len(clean_readings)


class TestCheckOutliersZscore:
    def test_no_outliers_in_clean_data(self, clean_readings):
        df = pd.DataFrame(clean_readings)
        outliers = check_outliers_zscore(df)
        assert all(len(v) == 0 for v in outliers.values())

    def test_detects_extreme_outlier(self, clean_readings):
        readings = list(clean_readings)
        readings.append({**clean_readings[0], "engine_temp_c": 9999.0})
        df = pd.DataFrame(readings)
        outliers = check_outliers_zscore(df, threshold=3.0)
        assert "engine_temp_c" in outliers


class TestAssessBatchQuality:
    def test_perfect_quality_score(self, clean_readings):
        result = assess_batch_quality(clean_readings)
        assert result["quality_score"] > 0.9

    def test_empty_returns_zero(self):
        result = assess_batch_quality([])
        assert result["quality_score"] == 0.0

    def test_has_required_keys(self, clean_readings):
        result = assess_batch_quality(clean_readings)
        for key in ("total_rows", "missing_values", "outliers", "quality_score"):
            assert key in result

    @pytest.mark.parametrize("n_rows", [1, 5, 20])
    def test_total_rows_matches_input(self, clean_readings, n_rows):
        result = assess_batch_quality(clean_readings[:n_rows])
        assert result["total_rows"] == n_rows

    def test_quality_score_between_zero_one(self, clean_readings):
        result = assess_batch_quality(clean_readings)
        assert 0.0 <= result["quality_score"] <= 1.0
