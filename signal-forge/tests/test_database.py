"""Tests for SQLAlchemy models and database session management."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from sqlalchemy import text

from app.database import DriftEvent, MarketRegimePrediction


def test_db_session_executes_query(db_session):
    result = db_session.execute(text("SELECT 1"))
    assert result.scalar() == 1


def test_create_prediction_record(db_session):
    record = MarketRegimePrediction(
        ticker="AAPL",
        regime="bull",
        confidence=0.85,
        risk_score=0.2,
    )
    db_session.add(record)
    db_session.commit()
    assert record.id is not None


def test_prediction_record_fields(db_session):
    record = MarketRegimePrediction(
        ticker="TSLA",
        regime="volatile",
        confidence=0.72,
        risk_score=0.78,
        volatility=0.35,
        momentum=0.08,
        correlation=0.5,
        volume_ratio=1.3,
        beta=1.6,
    )
    db_session.add(record)
    db_session.commit()
    fetched = db_session.query(MarketRegimePrediction).filter_by(id=record.id).first()
    assert fetched is not None
    assert fetched.regime == "volatile"
    assert fetched.volatility == pytest.approx(0.35)


def test_create_drift_event(db_session):
    event = DriftEvent(
        feature_name="volatility",
        ks_statistic=0.35,
        p_value=0.002,
        drift_detected=1,
    )
    db_session.add(event)
    db_session.commit()
    assert event.id is not None


@pytest.mark.parametrize("regime", ["bull", "bear", "sideways", "volatile"])
def test_create_prediction_all_regimes(db_session, regime):
    record = MarketRegimePrediction(
        ticker="SPY",
        regime=regime,
        confidence=0.6,
        risk_score=0.5,
    )
    db_session.add(record)
    db_session.commit()
    assert record.regime == regime
