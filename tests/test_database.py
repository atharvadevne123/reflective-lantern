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


def test_query_by_building_id(db_session):
    for i in range(3):
        db_session.add(
            EnergyReading(
                building_id="query-test-bldg",
                timestamp=datetime.utcnow(),
                consumption_kwh=10.0 + i,
            )
        )
    db_session.commit()
    rows = db_session.query(EnergyReading).filter(EnergyReading.building_id == "query-test-bldg").all()
    assert len(rows) >= 3
