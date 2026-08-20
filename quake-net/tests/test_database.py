"""Tests for SQLAlchemy models and persistence behaviour."""

from __future__ import annotations

from datetime import datetime

import pytest

from app.database import DriftLog, ModelMetrics, SeismicEvent


def _event(**overrides) -> SeismicEvent:
    payload = {
        "latitude": 35.6,
        "longitude": 139.7,
        "depth_km": 20.0,
        "station_count": 12,
        "p_wave_amplitude": 4.1,
        "s_wave_amplitude": 7.8,
        "epicentral_distance_km": 100.0,
        "fault_type": "reverse",
        "predicted_magnitude": 5.4,
        "aftershock_probability": 0.62,
    }
    payload.update(overrides)
    return SeismicEvent(**payload)


class TestSeismicEventModel:
    def test_insert_assigns_primary_key(self, db_session) -> None:
        event = _event()
        db_session.add(event)
        db_session.commit()
        assert event.id is not None

    def test_created_at_defaults_to_now(self, db_session) -> None:
        event = _event()
        db_session.add(event)
        db_session.commit()
        assert isinstance(event.created_at, datetime)

    def test_model_version_defaults(self, db_session) -> None:
        event = _event()
        db_session.add(event)
        db_session.commit()
        assert event.model_version == "1.0.0"

    def test_round_trip_preserves_values(self, db_session) -> None:
        db_session.add(_event(predicted_magnitude=6.7))
        db_session.commit()
        loaded = db_session.query(SeismicEvent).filter_by(predicted_magnitude=6.7).one()
        assert loaded.fault_type == "reverse"

    def test_query_ordering_by_recency(self, db_session) -> None:
        for magnitude in (3.0, 4.0, 5.0):
            db_session.add(_event(predicted_magnitude=magnitude))
        db_session.commit()
        rows = db_session.query(SeismicEvent).order_by(SeismicEvent.id.desc()).limit(3).all()
        assert rows[0].predicted_magnitude == 5.0

    def test_aftershock_probability_nullable(self, db_session) -> None:
        event = _event(aftershock_probability=None)
        db_session.add(event)
        db_session.commit()
        assert event.aftershock_probability is None

    @pytest.mark.parametrize("fault_type", ["strike_slip", "reverse", "normal", "oblique"])
    def test_all_fault_types_persist(self, db_session, fault_type: str) -> None:
        event = _event(fault_type=fault_type)
        db_session.add(event)
        db_session.commit()
        assert event.fault_type == fault_type


class TestDriftLogModel:
    def test_insert_and_read_back(self, db_session) -> None:
        log = DriftLog(
            feature_name="depth_km",
            ks_statistic=0.21,
            p_value=0.004,
            drift_detected=True,
            sample_size=200,
        )
        db_session.add(log)
        db_session.commit()
        assert log.id is not None

    def test_drift_flag_persisted(self, db_session) -> None:
        log = DriftLog(
            feature_name="p_wave_amplitude",
            ks_statistic=0.05,
            p_value=0.8,
            drift_detected=False,
            sample_size=150,
        )
        db_session.add(log)
        db_session.commit()
        loaded = db_session.query(DriftLog).filter_by(feature_name="p_wave_amplitude").one()
        assert loaded.drift_detected is False

    def test_checked_at_default(self, db_session) -> None:
        log = DriftLog(
            feature_name="station_count",
            ks_statistic=0.1,
            p_value=0.3,
            drift_detected=False,
            sample_size=50,
        )
        db_session.add(log)
        db_session.commit()
        assert isinstance(log.checked_at, datetime)


class TestModelMetricsModel:
    def test_insert_and_read_back(self, db_session) -> None:
        metrics = ModelMetrics(
            model_version="1.0.0",
            rmse=0.32,
            mae=0.25,
            r2=0.87,
            cv_r2_mean=0.85,
            cv_r2_std=0.02,
            n_features=44,
            n_samples=2000,
        )
        db_session.add(metrics)
        db_session.commit()
        assert metrics.id is not None

    def test_notes_field_optional(self, db_session) -> None:
        metrics = ModelMetrics(
            model_version="1.0.1",
            rmse=0.30,
            mae=0.24,
            r2=0.88,
            cv_r2_mean=0.86,
            cv_r2_std=0.02,
            n_features=44,
            n_samples=2500,
        )
        db_session.add(metrics)
        db_session.commit()
        assert metrics.notes is None

    def test_trained_at_default(self, db_session) -> None:
        metrics = ModelMetrics(
            model_version="1.0.2",
            rmse=0.29,
            mae=0.23,
            r2=0.89,
            cv_r2_mean=0.87,
            cv_r2_std=0.01,
            n_features=44,
            n_samples=3000,
        )
        db_session.add(metrics)
        db_session.commit()
        assert isinstance(metrics.trained_at, datetime)
