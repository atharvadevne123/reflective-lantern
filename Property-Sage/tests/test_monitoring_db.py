"""Tests for prediction_summary_stats and get_drift_history."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, DriftLog, Prediction
from app.monitoring import get_drift_history, prediction_summary_stats


@pytest.fixture(scope="module")
def db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_summary_stats_empty(db):
    result = prediction_summary_stats(db)
    assert result["count"] == 0


def test_summary_stats_with_data(db):
    for i in range(3):
        pred = Prediction(
            request_id=f"mdb-sumstat-{i}",
            bedrooms=3, bathrooms=2.0, sqft=1500.0,
            lot_size=5000.0, year_built=2005,
            neighborhood="suburb", property_type="house",
            predicted_price=400_000.0 + i * 10_000,
            predicted_rental_yield=0.05,
        )
        db.add(pred)
    db.commit()
    result = prediction_summary_stats(db)
    assert result["count"] >= 3
    assert result["mean_price"] > 0


def test_drift_history_empty(db):
    result = get_drift_history(db)
    assert isinstance(result, list)


def test_drift_history_after_insert(db):
    log = DriftLog(feature_name="price", ks_statistic=0.12, p_value=0.03, drift_detected=1)
    db.add(log)
    db.commit()
    history = get_drift_history(db, limit=10)
    assert len(history) >= 1
    assert "feature_name" in history[0]


@pytest.mark.parametrize("field", ["count", "mean_price", "mean_yield"])
def test_stats_keys(db, field):
    result = prediction_summary_stats(db)
    assert field in result
