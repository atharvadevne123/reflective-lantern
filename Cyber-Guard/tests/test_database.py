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


class TestPredictionLog:
    def test_insert_and_query(self, db_session: Session) -> None:
        log = PredictionLog(
            raw_features='{"duration":1}',
            predicted_class="normal",
            predicted_proba=0.95,
        )
        db_session.add(log)
        db_session.commit()
        result = db_session.query(PredictionLog).first()
        assert result is not None
        assert result.predicted_class == "normal"

    def test_multiple_inserts(self, db_session: Session) -> None:
        for i in range(5):
            db_session.add(
                PredictionLog(
                    raw_features=f'{{"idx":{i}}}',
                    predicted_class="normal" if i % 2 == 0 else "anomaly",
                    predicted_proba=0.5 + i * 0.1,
                )
            )
        db_session.commit()
        count = db_session.query(PredictionLog).count()
        assert count == 5

    def test_default_timestamps(self, db_session: Session) -> None:
        log = PredictionLog(
            raw_features="{}",
            predicted_class="normal",
            predicted_proba=0.9,
        )
        db_session.add(log)
        db_session.commit()
        result = db_session.query(PredictionLog).first()
        assert result is not None
