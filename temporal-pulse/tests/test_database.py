"""Tests for SQLAlchemy models and database session management."""

from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestSensorReading:
    def test_insert_and_query(self, db_session):
        from app.database import SensorReading

        reading = SensorReading(
            sensor_id="s1",
            timestamp=datetime(2026, 1, 1, 12, 0),
            values={"temp": 21.5, "humidity": 55.0},
        )
        db_session.add(reading)
        db_session.flush()
        found = db_session.query(SensorReading).filter_by(sensor_id="s1").first()
        assert found is not None
        assert found.values["temp"] == 21.5

    def test_created_at_auto_set(self, db_session):
        from app.database import SensorReading

        reading = SensorReading(sensor_id="s2", timestamp=datetime(2026, 1, 1), values={"x": 1.0})
        db_session.add(reading)
        db_session.flush()
        assert reading.created_at is not None


class TestAnomalyEvent:
    def test_insert_anomaly_event(self, db_session):
        from app.database import AnomalyEvent

        event = AnomalyEvent(
            sensor_id="s1",
            timestamp=datetime(2026, 1, 1),
            anomaly_score=0.95,
            root_cause="vibration spike",
            features_snapshot={"vibration": 12.3},
        )
        db_session.add(event)
        db_session.flush()
        found = db_session.query(AnomalyEvent).filter_by(sensor_id="s1").first()
        assert found.anomaly_score == 0.95

    def test_query_by_score_threshold(self, db_session):
        from app.database import AnomalyEvent

        for score in [0.3, 0.8, 0.9]:
            db_session.add(
                AnomalyEvent(sensor_id="s3", timestamp=datetime(2026, 1, 1), anomaly_score=score)
            )
        db_session.flush()
        high = (
            db_session.query(AnomalyEvent)
            .filter(AnomalyEvent.sensor_id == "s3", AnomalyEvent.anomaly_score >= 0.7)
            .count()
        )
        assert high == 2


class TestPrediction:
    def test_insert_prediction(self, db_session):
        from app.database import Prediction

        pred = Prediction(
            sensor_id="s1",
            horizon=5,
            predicted_values=[1.0, 2.0, 3.0, 4.0, 5.0],
            confidence_interval={"lower": [0.5], "upper": [1.5]},
            model_version="v1.0",
        )
        db_session.add(pred)
        db_session.flush()
        found = db_session.query(Prediction).filter_by(sensor_id="s1").first()
        assert len(found.predicted_values) == 5


class TestDriftLog:
    def test_insert_drift_log(self, db_session):
        from app.database import DriftLog

        log = DriftLog(
            sensor_id="s1",
            feature_name="temp_roll_mean_5",
            ks_statistic=0.42,
            p_value=0.001,
            drift_detected=1,
        )
        db_session.add(log)
        db_session.flush()
        found = db_session.query(DriftLog).filter_by(feature_name="temp_roll_mean_5").first()
        assert found.drift_detected == 1


class TestInitDb:
    def test_init_db_creates_tables(self):
        from app.database import init_db

        init_db()

    import pytest

    @pytest.mark.parametrize("drift_detected", [0, 1])
    def test_drift_log_detected_values(self, db_session, drift_detected: int) -> None:
        from app.database import DriftLog

        log = DriftLog(
            sensor_id="s2",
            feature_name=f"feature_{drift_detected}",
            ks_statistic=0.3,
            p_value=0.05,
            drift_detected=drift_detected,
        )
        db_session.add(log)
        db_session.flush()
        found = (
            db_session.query(DriftLog).filter_by(feature_name=f"feature_{drift_detected}").first()
        )
        assert found.drift_detected == drift_detected

    def test_drift_log_ks_statistic_stored(self, db_session) -> None:
        from app.database import DriftLog

        log = DriftLog(
            sensor_id="s3",
            feature_name="vibration_mean",
            ks_statistic=0.75,
            p_value=0.02,
            drift_detected=1,
        )
        db_session.add(log)
        db_session.flush()
        found = db_session.query(DriftLog).filter_by(feature_name="vibration_mean").first()
        assert abs(found.ks_statistic - 0.75) < 1e-6
