"""Tests for sport-cast app/database.py."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base, MatchRecord, PredictionLog


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
    def test_match_records_table(self, engine) -> None:
        assert "match_records" in inspect(engine).get_table_names()

    def test_prediction_logs_table(self, engine) -> None:
        assert "prediction_logs" in inspect(engine).get_table_names()


class TestMatchRecord:
    def test_insert(self, db_session: Session) -> None:
        rec = MatchRecord(
            match_id="m1",
            home_team="TeamA",
            away_team="TeamB",
            sport="football",
        )
        db_session.add(rec)
        db_session.commit()
        result = db_session.query(MatchRecord).first()
        assert result is not None
        assert result.home_team == "TeamA"

    def test_unique_match_id(self, db_session: Session) -> None:
        db_session.add(MatchRecord(match_id="unique1", home_team="A", away_team="B"))
        db_session.commit()
        count = db_session.query(MatchRecord).filter_by(match_id="unique1").count()
        assert count == 1


class TestPredictionLog:
    def test_insert(self, db_session: Session) -> None:
        log = PredictionLog(
            match_id="m1",
            predicted_winner="TeamA",
            confidence=0.75,
            model_version="1.0.0",
        )
        db_session.add(log)
        db_session.commit()
        result = db_session.query(PredictionLog).first()
        assert result is not None
        assert result.confidence == pytest.approx(0.75)
