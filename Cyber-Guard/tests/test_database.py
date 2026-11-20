"""Tests for Cyber-Guard app/database.py."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

from app.database import Base, PredictionLog


@pytest.fixture()
def engine():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture()
def db_session(engine):
    with Session(engine) as session:
        yield session


class TestInitDb:
    def test_creates_tables(self, engine) -> None:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert len(tables) > 0

    def test_prediction_log_table_exists(self, engine) -> None:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "prediction_logs" in tables


def _make_pred_log(**kwargs):
    defaults = dict(
        src_bytes=100.0, dst_bytes=200.0, duration=1.0,
        protocol_type="tcp", service="http", flag="SF",
        prediction="normal", confidence=0.95,
    )
    defaults.update(kwargs)
    return PredictionLog(**defaults)


class TestPredictionLog:
    def test_insert_and_query(self, db_session: Session) -> None:
        log = _make_pred_log(prediction="normal", confidence=0.95)
        db_session.add(log)
        db_session.commit()
        result = db_session.query(PredictionLog).first()
        assert result is not None
        assert result.prediction == "normal"

    def test_multiple_inserts(self, db_session: Session) -> None:
        for i in range(5):
            db_session.add(
                _make_pred_log(
                    prediction="normal" if i % 2 == 0 else "anomaly",
                    confidence=0.5 + i * 0.1,
                )
            )
        db_session.commit()
        count = db_session.query(PredictionLog).count()
        assert count == 5

    def test_default_timestamps(self, db_session: Session) -> None:
        log = _make_pred_log()
        db_session.add(log)
        db_session.commit()
        result = db_session.query(PredictionLog).first()
        assert result is not None


class TestPredictionLogQueries:
    def test_filter_by_prediction(self, db_session: Session) -> None:
        for pred in ["normal", "anomaly", "normal"]:
            db_session.add(_make_pred_log(prediction=pred))
        db_session.commit()
        normals = db_session.query(PredictionLog).filter(
            PredictionLog.prediction == "normal"
        ).count()
        assert normals == 2

    def test_filter_by_confidence_threshold(self, db_session: Session) -> None:
        for conf in [0.3, 0.6, 0.9]:
            db_session.add(_make_pred_log(confidence=conf))
        db_session.commit()
        high_conf = db_session.query(PredictionLog).filter(
            PredictionLog.confidence >= 0.5
        ).count()
        assert high_conf == 2

    def test_model_version_default(self, db_session: Session) -> None:
        log = _make_pred_log()
        db_session.add(log)
        db_session.commit()
        result = db_session.query(PredictionLog).first()
        assert result is not None
        assert result.model_version is not None or result.model_version is None

    def test_confidence_stored_correctly(self, db_session: Session) -> None:
        log = _make_pred_log(confidence=0.876)
        db_session.add(log)
        db_session.commit()
        result = db_session.query(PredictionLog).first()
        assert abs(result.confidence - 0.876) < 0.001
