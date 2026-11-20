"""Tests for database helper functions."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, Prediction, get_prediction_count, get_recent_predictions


@pytest.fixture(scope="module")
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_prediction_count_empty(db_session):
    assert get_prediction_count(db_session) == 0


def test_prediction_count_after_insert(db_session):
    pred = Prediction(
        request_id="test-1",
        bedrooms=3, bathrooms=2.0, sqft=1500.0,
        lot_size=5000.0, year_built=2005,
        neighborhood="suburb", property_type="house",
        predicted_price=400000.0, predicted_rental_yield=0.05,
    )
    db_session.add(pred)
    db_session.commit()
    assert get_prediction_count(db_session) >= 1


def test_get_recent_predictions_returns_list(db_session):
    result = get_recent_predictions(db_session, limit=5)
    assert isinstance(result, list)


@pytest.mark.parametrize("limit", [1, 5, 10])
def test_get_recent_predictions_limit(db_session, limit):
    result = get_recent_predictions(db_session, limit=limit)
    assert len(result) <= limit
