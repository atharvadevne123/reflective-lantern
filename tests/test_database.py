"""Tests for database models and session management."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, DriftReport, NeighborhoodStat, PredictionLog, Property

TEST_URL = "sqlite:///:memory:"


@pytest.fixture(scope="module")
def mem_db() -> None:
    engine = create_engine(TEST_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    yield db
    db.close()
    Base.metadata.drop_all(bind=engine)


def test_create_prediction_log(mem_db) -> None:
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


def test_create_drift_report(mem_db) -> None:
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


def test_create_neighborhood_stat(mem_db) -> None:
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


def test_query_prediction_logs_by_correlation(mem_db) -> None:
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


def test_create_property(mem_db) -> None:
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


def test_drift_report_no_drift(mem_db) -> None:
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


@pytest.mark.parametrize(
    "predicted_value",
    [100_000.0, 500_000.0, 1_200_000.0, 5_000_000.0],
)
def test_prediction_log_various_values(mem_db, predicted_value: float) -> None:
    import math

    log = PredictionLog(
        predicted_value=predicted_value,
        model_version="1.0.0",
        features_json="{}",
        correlation_id=f"val-{int(predicted_value)}",
    )
    mem_db.add(log)
    mem_db.commit()
    assert log.id is not None
    assert math.isclose(log.predicted_value, predicted_value)


def test_multiple_drift_reports_for_same_feature(mem_db) -> None:
    for i in range(3):
        report = DriftReport(
            feature_name="sqft",
            ks_statistic=0.1 * (i + 1),
            p_value=0.05 * (i + 1),
            drift_detected=i > 1,
            sample_size=50 + i * 10,
        )
        mem_db.add(report)
    mem_db.commit()
    sqft_reports = mem_db.query(DriftReport).filter(DriftReport.feature_name == "sqft").all()
    assert len(sqft_reports) >= 3


def test_prediction_log_features_json_round_trip(mem_db) -> None:
    import json

    features = {"sqft": 2000, "bedrooms": 4, "bathrooms": 3.0, "zipcode": "94103"}
    log = PredictionLog(
        predicted_value=800_000.0,
        model_version="1.0.0",
        features_json=json.dumps(features),
        correlation_id="json-rt-001",
    )
    mem_db.add(log)
    mem_db.commit()
    restored = json.loads(log.features_json)
    assert restored["sqft"] == 2000
    assert restored["bedrooms"] == 4


@pytest.mark.parametrize(
    "feature_name,ks_stat,p_val",
    [
        ("sqft", 0.1, 0.3),
        ("bedrooms", 0.05, 0.6),
        ("year_built", 0.2, 0.01),
    ],
)
def test_drift_report_various_features(mem_db, feature_name, ks_stat, p_val) -> None:
    report = DriftReport(
        feature_name=feature_name,
        ks_statistic=ks_stat,
        p_value=p_val,
        drift_detected=p_val < 0.05,
        sample_size=100,
    )
    mem_db.add(report)
    mem_db.commit()
    assert report.id is not None
    assert report.feature_name == feature_name


def test_neighborhood_stat_zipcode_stored(mem_db) -> None:
    stat = NeighborhoodStat(
        zipcode="10001",
        median_price=750_000.0,
        median_price_per_sqft=600.0,
        school_score=7.5,
        transit_score=9.0,
        walkability_score=9.5,
        crime_rate=0.2,
        avg_rental_yield=0.04,
    )
    mem_db.add(stat)
    mem_db.commit()
    found = mem_db.query(NeighborhoodStat).filter(NeighborhoodStat.zipcode == "10001").first()
    assert found is not None
    assert found.median_price == 750_000.0
