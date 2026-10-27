"""Tests for trade-sage app/database.py."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker

import app.database as db_module
from app.database import Base, PredictionLog


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
        inspector = inspect(engine)
        assert "prediction_logs" in inspector.get_table_names()


class TestPredictionLog:
    def test_insert(self, db_session: Session) -> None:
        log = PredictionLog(
            ticker="AAPL",
            prediction=1,
            confidence=0.85,
            model_version="1.0.0",
        )
        db_session.add(log)
        db_session.commit()
        result = db_session.query(PredictionLog).first()
        assert result is not None
        assert result.ticker == "AAPL"

    def test_query_by_ticker(self, db_session: Session) -> None:
        for ticker in ["AAPL", "GOOG", "AAPL"]:
            db_session.add(PredictionLog(ticker=ticker, prediction=1, confidence=0.8, model_version="1"))
        db_session.commit()
        appl = db_session.query(PredictionLog).filter_by(ticker="AAPL").count()
        assert appl == 2

    @pytest.mark.parametrize("prediction", [0, 1])
    def test_binary_prediction_values(self, db_session: Session, prediction: int) -> None:
        log = PredictionLog(ticker="TEST", prediction=prediction, confidence=0.5, model_version="1")
        db_session.add(log)
        db_session.commit()
        result = db_session.query(PredictionLog).filter_by(ticker="TEST").first()
        assert result is not None
        assert result.prediction == prediction
