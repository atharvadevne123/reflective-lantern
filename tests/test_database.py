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


def test_prediction_log_actual_kwh_nullable(db_session):
    p = PredictionLog(
        building_id="nullable-test",
        timestamp=datetime.utcnow(),
        predicted_kwh=10.0,
        latency_ms=5.0,
        actual_kwh=None,
    )
    db_session.add(p)
    db_session.commit()
    assert p.actual_kwh is None


def test_drift_log_drift_detected_zero(db_session):
    d = DriftLog(
        feature_name="temperature_c",
        ks_statistic=0.02,
        p_value=0.85,
        drift_detected=0,
    )
    db_session.add(d)
    db_session.commit()
    fetched = db_session.query(DriftLog).filter(DriftLog.feature_name == "temperature_c").first()
    assert fetched is not None
    assert fetched.drift_detected == 0


def test_anomaly_log_severity_none(db_session):
    a = AnomalyLog(
        building_id="normal-bldg",
        timestamp=datetime.utcnow(),
        consumption_kwh=12.0,
        anomaly_score=0.1,
        is_anomaly=0,
        severity="none",
    )
    db_session.add(a)
    db_session.commit()
    assert a.severity == "none"


def test_get_predictions_by_building_empty(db_session):
    from app.database import get_predictions_by_building

    results = get_predictions_by_building(db_session, "nonexistent-bldg")
    assert results == []


def test_get_predictions_by_building_filters(db_session):
    import uuid
    from datetime import datetime

    from app.database import PredictionLog, get_predictions_by_building

    uid = uuid.uuid4().hex[:8]
    bid_a = f"filter-A-{uid}"
    bid_b = f"filter-B-{uid}"
    for bid in [bid_a, bid_a, bid_b]:
        entry = PredictionLog(
            building_id=bid,
            timestamp=datetime.utcnow(),
            predicted_kwh=10.0,
            latency_ms=5.0,
        )
        db_session.add(entry)
    db_session.commit()

    results_a = get_predictions_by_building(db_session, bid_a)
    results_b = get_predictions_by_building(db_session, bid_b)
    assert len(results_a) == 2
    assert len(results_b) == 1


def test_get_predictions_by_building_limit(db_session):
    from datetime import datetime

    from app.database import PredictionLog, get_predictions_by_building

    for i in range(5):
        entry = PredictionLog(
            building_id="bldg-limit",
            timestamp=datetime.utcnow(),
            predicted_kwh=float(i),
            latency_ms=1.0,
        )
        db_session.add(entry)
    db_session.commit()
    results = get_predictions_by_building(db_session, "bldg-limit", limit=3)
    assert len(results) == 3


def test_prediction_log_missing_building_returns_empty(db_session):
    from app.database import get_predictions_by_building

    results = get_predictions_by_building(db_session, "nonexistent-building")
    assert results == []


def test_anomaly_log_stored_and_retrieved(db_session):
    from datetime import datetime

    from app.database import AnomalyLog

    entry = AnomalyLog(
        building_id="bldg-anomaly",
        timestamp=datetime.utcnow(),
        consumption_kwh=95.0,
        anomaly_score=0.85,
        is_anomaly=True,
        severity="critical",
    )
    db_session.add(entry)
    db_session.commit()
    retrieved = db_session.query(AnomalyLog).filter(AnomalyLog.building_id == "bldg-anomaly").first()
    assert retrieved is not None
    assert retrieved.severity == "critical"


def test_drift_log_stored_and_retrieved(db_session):
    from datetime import datetime

    from app.database import DriftLog

    entry = DriftLog(
        feature_name="consumption_kwh",
        ks_statistic=0.42,
        p_value=0.02,
        drift_detected=True,
        checked_at=datetime.utcnow(),
    )
    db_session.add(entry)
    db_session.commit()
    retrieved = db_session.query(DriftLog).filter(DriftLog.feature_name == "consumption_kwh").first()
    assert retrieved is not None
    assert bool(retrieved.drift_detected) is True


@pytest.mark.parametrize("n", [1, 3, 5, 10])
def test_prediction_log_multiple_buildings(db_session, n):
    from datetime import datetime

    from app.database import PredictionLog, get_predictions_by_building

    for i in range(n):
        db_session.add(
            PredictionLog(
                building_id=f"building-{i}",
                timestamp=datetime.utcnow(),
                predicted_kwh=float(i * 2),
                latency_ms=float(i),
            )
        )
    db_session.commit()
    for i in range(n):
        results = get_predictions_by_building(db_session, f"building-{i}")
        assert len(results) == 1
        assert results[0].predicted_kwh == float(i * 2)


def test_get_recent_anomalies_empty(db_session):
    from app.database import get_recent_anomalies

    results = get_recent_anomalies(db_session, "nonexistent-building-xyz")
    assert results == []


def test_get_recent_anomalies_returns_records(db_session):
    from datetime import datetime

    from app.database import AnomalyLog, get_recent_anomalies

    for i in range(3):
        db_session.add(
            AnomalyLog(
                building_id="anomaly-query-bldg",
                timestamp=datetime.utcnow(),
                consumption_kwh=90.0 + i,
                anomaly_score=0.9,
                is_anomaly=1,
                severity="critical",
            )
        )
    db_session.commit()
    results = get_recent_anomalies(db_session, "anomaly-query-bldg")
    assert len(results) >= 3


def test_get_recent_anomalies_severity_filter(db_session):
    from datetime import datetime

    from app.database import AnomalyLog, get_recent_anomalies

    db_session.add(
        AnomalyLog(
            building_id="severity-bldg",
            timestamp=datetime.utcnow(),
            consumption_kwh=80.0,
            anomaly_score=0.8,
            is_anomaly=1,
            severity="low",
        )
    )
    db_session.add(
        AnomalyLog(
            building_id="severity-bldg",
            timestamp=datetime.utcnow(),
            consumption_kwh=95.0,
            anomaly_score=0.95,
            is_anomaly=1,
            severity="critical",
        )
    )
    db_session.commit()
    critical = get_recent_anomalies(db_session, "severity-bldg", severity="critical")
    assert all(r.severity == "critical" for r in critical)


def test_count_anomalies_by_building_zero(db_session):
    from app.database import count_anomalies_by_building

    count = count_anomalies_by_building(db_session, "no-such-building-count")
    assert count == 0


def test_count_anomalies_by_building_positive(db_session):
    from datetime import datetime

    from app.database import AnomalyLog, count_anomalies_by_building

    for _ in range(4):
        db_session.add(
            AnomalyLog(
                building_id="count-bldg",
                timestamp=datetime.utcnow(),
                consumption_kwh=88.0,
                anomaly_score=0.88,
                is_anomaly=1,
                severity="medium",
            )
        )
    db_session.commit()
    assert count_anomalies_by_building(db_session, "count-bldg") == 4


@pytest.mark.parametrize("limit", [1, 2, 5])
def test_get_recent_anomalies_limit(db_session, limit):
    from datetime import datetime

    from app.database import AnomalyLog, get_recent_anomalies

    for i in range(6):
        db_session.add(
            AnomalyLog(
                building_id="limit-anomaly-bldg",
                timestamp=datetime.utcnow(),
                consumption_kwh=float(90 + i),
                anomaly_score=0.9,
                is_anomaly=1,
                severity="critical",
            )
        )
    db_session.commit()
    results = get_recent_anomalies(db_session, "limit-anomaly-bldg", limit=limit)
    assert len(results) == limit
