"""Database model tests."""

from __future__ import annotations

from datetime import datetime

import pytest

from app.database import AnomalyLog, DriftLog, EnergyReading, PredictionLog


def test_energy_reading_insert(db_session):
    r = EnergyReading(
        building_id="bldg-db-test",
        timestamp=datetime.utcnow(),
        consumption_kwh=15.2,
        temperature_c=22.0,
        humidity_pct=55.0,
        occupancy=30,
        hvac_state=1,
    )
    db_session.add(r)
    db_session.commit()
    assert r.id is not None


def test_prediction_log_insert(db_session):
    p = PredictionLog(
        building_id="bldg-db-test",
        timestamp=datetime.utcnow(),
        predicted_kwh=18.5,
        latency_ms=12.3,
    )
    db_session.add(p)
    db_session.commit()
    assert p.id is not None
    assert p.model_version == "1.0.0"


def test_anomaly_log_insert(db_session):
    a = AnomalyLog(
        building_id="bldg-db-test",
        timestamp=datetime.utcnow(),
        consumption_kwh=99.0,
        anomaly_score=-0.7,
        is_anomaly=1,
        severity="critical",
    )
    db_session.add(a)
    db_session.commit()
    assert a.id is not None


def test_drift_log_insert(db_session):
    d = DriftLog(
        feature_name="consumption_kwh",
        ks_statistic=0.42,
        p_value=0.001,
        drift_detected=1,
    )
    db_session.add(d)
    db_session.commit()
    assert d.id is not None


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


def test_address_max_len_constant() -> None:
    from app.database import ADDRESS_MAX_LEN

    assert ADDRESS_MAX_LEN > 0


def test_zipcode_max_len_constant() -> None:
    from app.database import ZIPCODE_MAX_LEN

    assert ZIPCODE_MAX_LEN > 0


def test_model_version_max_len_constant() -> None:
    from app.database import MODEL_VERSION_MAX_LEN

    assert MODEL_VERSION_MAX_LEN > 0


def test_feature_name_max_len_constant() -> None:
    from app.database import FEATURE_NAME_MAX_LEN

    assert FEATURE_NAME_MAX_LEN > 0


def test_correlation_id_max_len_constant() -> None:
    from app.database import CORRELATION_ID_MAX_LEN

    assert CORRELATION_ID_MAX_LEN > 0


@pytest.mark.parametrize("const_name,expected_min", [
    ("ADDRESS_MAX_LEN", 100),
    ("ZIPCODE_MAX_LEN", 5),
    ("CITY_MAX_LEN", 10),
    ("STATE_MAX_LEN", 2),
    ("MODEL_VERSION_MAX_LEN", 10),
])
def test_db_constants_meet_minimum_lengths(const_name: str, expected_min: int) -> None:
    import app.database as db_module

    assert getattr(db_module, const_name) >= expected_min
