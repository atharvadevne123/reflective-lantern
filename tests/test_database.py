"""Tests for database models and session management."""

from __future__ import annotations

from app.database import DriftReport, EnergyReading, PredictionLog, RetrainingLog


class TestEnergyReadingModel:
    def test_create_energy_reading(self, db_session):
        reading = EnergyReading(
            load_mw=4500.0,
            temperature_c=28.0,
            humidity_pct=65.0,
            hour=14,
            day_of_week=2,
            month=7,
            is_weekend=False,
            region="northeast",
        )
        db_session.add(reading)
        db_session.commit()
        assert reading.id is not None

    def test_energy_reading_defaults(self, db_session):
        reading = EnergyReading(load_mw=3000.0, hour=8, day_of_week=0, month=3)
        db_session.add(reading)
        db_session.commit()
        assert reading.region == "default"
        assert reading.source == "api"

    def test_multiple_readings(self, db_session):
        for i in range(5):
            db_session.add(EnergyReading(
                load_mw=3000.0 + i * 100,
                hour=i,
                day_of_week=0,
                month=1,
            ))
        db_session.commit()
        count = db_session.query(EnergyReading).count()
        assert count >= 5


class TestPredictionLogModel:
    def test_create_prediction_log(self, db_session):
        log = PredictionLog(
            predicted_load_mw=4200.0,
            region="west",
            latency_ms=12.5,
            request_id="test-req-001",
        )
        db_session.add(log)
        db_session.commit()
        assert log.id is not None

    def test_prediction_log_defaults(self, db_session):
        log = PredictionLog(predicted_load_mw=3500.0)
        db_session.add(log)
        db_session.commit()
        assert log.model_version == "1.0.0"
        assert log.region == "default"


class TestDriftReportModel:
    def test_create_drift_report(self, db_session):
        report = DriftReport(
            feature_name="temperature_c",
            ks_statistic=0.12,
            p_value=0.04,
            drift_detected=True,
        )
        db_session.add(report)
        db_session.commit()
        assert report.id is not None
        assert report.drift_detected is True

    def test_drift_report_defaults(self, db_session):
        report = DriftReport(
            feature_name="load_mw",
            ks_statistic=0.05,
            p_value=0.30,
            drift_detected=False,
        )
        db_session.add(report)
        db_session.commit()
        assert report.reference_window == 1000
        assert report.current_window == 200


class TestRetrainingLogModel:
    def test_create_retraining_log(self, db_session):
        log = RetrainingLog(
            trigger="drift_alert",
            samples_used=2000,
            r2_after=0.87,
            rmse_after=250.5,
            status="success",
        )
        db_session.add(log)
        db_session.commit()
        assert log.id is not None
        assert log.status == "success"
