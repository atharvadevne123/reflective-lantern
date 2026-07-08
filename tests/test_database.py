"""Tests for database models and session management."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, DriftReport, NeighborhoodStat, PredictionLog, Property

TEST_URL = "sqlite:///:memory:"


@pytest.fixture(scope="module")
def mem_db():
    engine = create_engine(TEST_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    yield db
    db.close()
    Base.metadata.drop_all(bind=engine)


def test_create_prediction_log(mem_db):
    log = PredictionLog(
        predicted_value=450_000.0,
        model_version="1.0.0",
        features_json='{"sqft": 1800}',
        correlation_id="test-001",
    )
    mem_db.add(log)
    mem_db.commit()
    assert log.id is not None
    assert log.predicted_value == 450_000.0


def test_create_drift_report(mem_db):
    report = DriftReport(
        feature_name="sqft",
        ks_statistic=0.15,
        p_value=0.03,
        drift_detected=True,
        sample_size=100,
    )
    mem_db.add(report)
    mem_db.commit()
    assert report.id is not None
    assert report.drift_detected is True


def test_create_neighborhood_stat(mem_db):
    stat = NeighborhoodStat(
        zipcode="94102",
        median_price=1_200_000.0,
        median_price_per_sqft=800.0,
        school_score=8.0,
        transit_score=9.0,
        walkability_score=8.5,
        crime_rate=0.25,
        avg_rental_yield=0.05,
    )
    mem_db.add(stat)
    mem_db.commit()
    assert stat.id is not None


def test_query_prediction_logs_by_correlation(mem_db):
    log = PredictionLog(
        predicted_value=300_000.0,
        model_version="1.0.0",
        features_json="{}",
        correlation_id="unique-corr-xyz",
    )
    mem_db.add(log)
    mem_db.commit()
    results = (
        mem_db.query(PredictionLog).filter(PredictionLog.correlation_id == "unique-corr-xyz").all()
    )
    assert len(results) == 1
    assert results[0].predicted_value == 300_000.0


def test_create_property(mem_db):
    prop = Property(
        address="123 Main St",
        zipcode="94102",
        city="San Francisco",
        state="CA",
        sqft=1800.0,
        bedrooms=3,
        bathrooms=2.0,
        lot_size=5000.0,
        year_built=1990,
        condition_score=7.5,
    )
    mem_db.add(prop)
    mem_db.commit()
    assert prop.id is not None


def test_drift_report_no_drift(mem_db):
    report = DriftReport(
        feature_name="predicted_value",
        ks_statistic=0.05,
        p_value=0.42,
        drift_detected=False,
        sample_size=200,
    )
    mem_db.add(report)
    mem_db.commit()
    assert report.drift_detected is False
