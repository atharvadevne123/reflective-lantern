"""Tests for threat-lens app/database.py."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base, DriftReport, PredictionLog


@pytest.fixture()
def engine():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture()
def db_session(engine):
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    with factory() as session:
        yield session


class TestSchema:
    def test_prediction_logs_table(self, engine) -> None:
        assert "prediction_logs" in inspect(engine).get_table_names()

    def test_drift_reports_table(self, engine) -> None:
        assert "drift_reports" in inspect(engine).get_table_names()


class TestPredictionLog:
    def test_insert(self, db_session: Session) -> None:
        log = PredictionLog(
            predicted_class="normal",
            confidence=0.92,
            is_attack=0,
        )
        db_session.add(log)
        db_session.commit()
        result = db_session.query(PredictionLog).first()
        assert result is not None
        assert result.predicted_class == "normal"

    def test_is_attack_values(self, db_session: Session) -> None:
        for is_attack in [0, 1]:
            db_session.add(
                PredictionLog(predicted_class="c", confidence=0.5, is_attack=is_attack)
            )
        db_session.commit()
        assert db_session.query(PredictionLog).count() == 2


class TestDriftReport:
    def test_insert(self, db_session: Session) -> None:
        report = DriftReport(
            feature_name="src_bytes",
            ks_statistic=0.12,
            p_value=0.03,
            drift_detected=True,
        )
        db_session.add(report)
        db_session.commit()
        result = db_session.query(DriftReport).first()
        assert result is not None
        assert result.feature_name == "src_bytes"
