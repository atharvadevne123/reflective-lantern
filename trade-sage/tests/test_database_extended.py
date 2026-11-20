"""Extended tests for database helper functions."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import (
    Base,
    DriftLog,
    PredictionLog,
    count_drift_alerts,
    count_predictions,
)


@pytest.fixture(scope="function")
def db_session():
    """Create an in-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


class TestCountPredictions:
    def test_empty_db_returns_zero(self, db_session) -> None:
        assert count_predictions(db_session) == 0

    def test_counts_inserted_rows(self, db_session) -> None:
        db_session.add(PredictionLog(ticker="AAPL", prediction=1, confidence=0.8, horizon_days=5))
        db_session.add(PredictionLog(ticker="MSFT", prediction=0, confidence=0.7, horizon_days=5))
        db_session.commit()
        assert count_predictions(db_session) == 2

    def test_filter_by_ticker(self, db_session) -> None:
        db_session.add(PredictionLog(ticker="AAPL", prediction=1, confidence=0.8, horizon_days=5))
        db_session.add(PredictionLog(ticker="GOOG", prediction=1, confidence=0.9, horizon_days=5))
        db_session.commit()
        assert count_predictions(db_session, ticker="AAPL") == 1
        assert count_predictions(db_session, ticker="GOOG") == 1

    def test_filter_returns_zero_for_missing_ticker(self, db_session) -> None:
        db_session.add(PredictionLog(ticker="AAPL", prediction=1, confidence=0.8, horizon_days=5))
        db_session.commit()
        assert count_predictions(db_session, ticker="TSLA") == 0

    def test_no_ticker_filter_counts_all(self, db_session) -> None:
        for _i in range(5):
            db_session.add(PredictionLog(ticker="AMZN", prediction=1, confidence=0.75, horizon_days=5))
        db_session.commit()
        assert count_predictions(db_session) == 5


class TestCountDriftAlerts:
    def test_empty_db_returns_zero(self, db_session) -> None:
        assert count_drift_alerts(db_session) == 0

    def test_counts_only_detected_drifts(self, db_session) -> None:
        db_session.add(DriftLog(feature_name="rsi", ks_statistic=0.4, p_value=0.01, drift_detected=True))
        db_session.add(DriftLog(feature_name="macd", ks_statistic=0.1, p_value=0.6, drift_detected=False))
        db_session.commit()
        assert count_drift_alerts(db_session) == 1

    def test_multiple_detected_drifts(self, db_session) -> None:
        for i in range(3):
            db_session.add(DriftLog(
                feature_name=f"f{i}", ks_statistic=0.5, p_value=0.01, drift_detected=True
            ))
        db_session.commit()
        assert count_drift_alerts(db_session) == 3
