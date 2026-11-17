"""Tests for Energy-Seer SQLAlchemy ORM models."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.database as db_module
from app.database import AnomalyLog, Base, DriftLog, EnergyReading, PredictionLog


@pytest.fixture(autouse=True)
def in_memory_db(tmp_path):
    test_url = "sqlite:///:memory:"
    engine = create_engine(test_url, connect_args={"check_same_thread": False})
    with patch.object(db_module, "engine", engine):
        with patch.object(db_module, "SessionLocal", sessionmaker(autocommit=False, autoflush=False, bind=engine)):
            Base.metadata.create_all(bind=engine)
            yield engine
    Base.metadata.drop_all(bind=engine)


def test_energy_reading_tablename():
    assert EnergyReading.__tablename__ == "energy_readings"


def test_prediction_log_tablename():
    assert PredictionLog.__tablename__ == "prediction_logs"


def test_anomaly_log_tablename():
    assert AnomalyLog.__tablename__ == "anomaly_logs"


def test_drift_log_tablename():
    assert DriftLog.__tablename__ == "drift_logs"


def test_energy_reading_columns(in_memory_db):
    cols = {c.name for c in EnergyReading.__table__.columns}
    assert "id" in cols
    assert "meter_id" in cols
    assert "consumption_kwh" in cols
    assert "building_type" in cols


def test_prediction_log_columns():
    cols = {c.name for c in PredictionLog.__table__.columns}
    assert "id" in cols
    assert "meter_id" in cols
    assert "predicted_kwh" in cols
    assert "model_version" in cols


def test_anomaly_log_columns():
    cols = {c.name for c in AnomalyLog.__table__.columns}
    assert "id" in cols
    assert "anomaly_score" in cols
    assert "is_anomaly" in cols
    assert "severity" in cols


def test_drift_log_columns():
    cols = {c.name for c in DriftLog.__table__.columns}
    assert "id" in cols
    assert "feature_name" in cols
    assert "ks_statistic" in cols
    assert "drift_detected" in cols


def test_insert_energy_reading(in_memory_db):
    Session = sessionmaker(bind=in_memory_db)
    session = Session()
    reading = EnergyReading(meter_id="meter1", consumption_kwh=42.5, building_type="office")
    session.add(reading)
    session.commit()
    result = session.query(EnergyReading).filter_by(meter_id="meter1").first()
    assert result is not None
    assert result.consumption_kwh == 42.5
    session.close()


def test_insert_anomaly_log(in_memory_db):
    Session = sessionmaker(bind=in_memory_db)
    session = Session()
    log = AnomalyLog(meter_id="m2", consumption_kwh=100.0, anomaly_score=0.9, is_anomaly=True, severity="high")
    session.add(log)
    session.commit()
    result = session.query(AnomalyLog).filter_by(meter_id="m2").first()
    assert result is not None
    assert result.is_anomaly is True
    session.close()
