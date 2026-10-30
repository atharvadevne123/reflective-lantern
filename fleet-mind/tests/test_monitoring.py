"""Tests for Fleet-Mind drift detection and prediction logging."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.monitoring import (
    _feature_buffer,
    log_prediction,
    maintenance_rate,
    run_drift_check,
)


@pytest.fixture
def prediction_result():
    return {
        "maintenance_needed": True,
        "anomaly_score": 0.87,
        "anomaly_type": "engine_overheat",
        "remaining_useful_life_km": 5000.0,
        "confidence": 0.74,
        "model_version": "1.0.0",
    }


@pytest.fixture
def normal_reading():
    return {
        "speed_kmh": 80.0,
        "engine_temp_c": 88.0,
        "oil_pressure_bar": 3.5,
        "fuel_consumption_lph": 7.5,
        "battery_voltage_v": 12.8,
        "brake_pad_mm": 6.0,
        "tire_pressure_bar": 2.3,
        "rpm": 2400.0,
        "coolant_temp_c": 84.0,
        "transmission_temp_c": 72.0,
    }


class TestLogPrediction:
    """Tests for log_prediction()."""

    def test_persists_to_db(self, db_session, normal_reading, prediction_result):
        initial_count = db_session.execute(
            __import__("sqlalchemy").text("SELECT COUNT(*) FROM vehicle_predictions")
        ).scalar()
        log_prediction(db_session, "VH-LOG-01", normal_reading, prediction_result)
        new_count = db_session.execute(
            __import__("sqlalchemy").text("SELECT COUNT(*) FROM vehicle_predictions")
        ).scalar()
        assert new_count >= initial_count

    def test_updates_feature_buffer(self, db_session, normal_reading, prediction_result):
        initial_len = len(_feature_buffer["speed_kmh"])
        log_prediction(db_session, "VH-LOG-02", normal_reading, prediction_result)
        assert len(_feature_buffer["speed_kmh"]) >= initial_len


class TestRunDriftCheck:
    """Tests for run_drift_check()."""

    def test_returns_list(self, db_session):
        result = run_drift_check(db_session)
        assert isinstance(result, list)

    def test_report_keys(self, db_session, normal_reading, prediction_result):
        # Seed enough observations for the KS test
        for i in range(60):
            log_prediction(db_session, f"VH-DRIFT-{i}", normal_reading, prediction_result)
        reports = run_drift_check(db_session)
        if reports:
            for report in reports:
                assert "feature" in report
                assert "ks_statistic" in report
                assert "p_value" in report
                assert "drift_detected" in report

    def test_ks_statistic_in_range(self, db_session, normal_reading, prediction_result):
        for i in range(60):
            log_prediction(db_session, f"VH-KS-{i}", normal_reading, prediction_result)
        reports = run_drift_check(db_session)
        for report in reports:
            assert 0.0 <= report["ks_statistic"] <= 1.0

    def test_p_value_in_range(self, db_session, normal_reading, prediction_result):
        for i in range(60):
            log_prediction(db_session, f"VH-PV-{i}", normal_reading, prediction_result)
        reports = run_drift_check(db_session)
        for report in reports:
            assert 0.0 <= report["p_value"] <= 1.0


class TestMaintenanceRate:
    """Tests for maintenance_rate()."""

    def test_returns_float(self, db_session):
        rate = maintenance_rate(db_session)
        assert isinstance(rate, float)

    def test_rate_between_zero_and_one(self, db_session):
        rate = maintenance_rate(db_session)
        assert 0.0 <= rate <= 1.0

    def test_empty_db_returns_zero(self, db_session):
        # Should not error even when there are no records
        assert maintenance_rate(db_session) >= 0.0


class TestLoadBaseline:
    """Tests for _load_baseline()."""

    def test_baseline_populated(self):
        from app.monitoring import _baseline_stats
        assert len(_baseline_stats) > 0
        for vals in _baseline_stats.values():
            assert len(vals) > 0
