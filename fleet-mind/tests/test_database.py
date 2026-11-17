"""Tests for Fleet-Mind SQLAlchemy ORM models."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.database as db_module
from app.database import Base, DriftReport, VehiclePrediction


@pytest.fixture(autouse=True)
def in_memory_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    monkeypatch.setattr(db_module, "_engine", engine)
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    monkeypatch.setattr(db_module, "_engine", None)


def test_vehicle_prediction_tablename():
    assert VehiclePrediction.__tablename__ == "vehicle_predictions"


def test_drift_report_tablename():
    assert DriftReport.__tablename__ == "drift_reports"


def test_vehicle_prediction_has_required_columns():
    cols = {c.name for c in VehiclePrediction.__table__.columns}
    assert "id" in cols
    assert "vehicle_id" in cols
    assert "speed_kmh" in cols
    assert "maintenance_needed" in cols
    assert "model_version" in cols


def test_drift_report_has_required_columns():
    cols = {c.name for c in DriftReport.__table__.columns}
    assert "id" in cols
    assert "feature_name" in cols
    assert "ks_statistic" in cols
    assert "drift_detected" in cols


def test_insert_vehicle_prediction(in_memory_db):
    Session = sessionmaker(bind=in_memory_db)
    session = Session()
    pred = VehiclePrediction(
        vehicle_id="v001",
        speed_kmh=60.0,
        engine_temp_c=90.0,
        oil_pressure_bar=3.5,
        fuel_consumption_lph=8.0,
        battery_voltage_v=12.5,
        brake_pad_mm=10.0,
        tire_pressure_bar=2.5,
        rpm=2000.0,
        coolant_temp_c=85.0,
        transmission_temp_c=70.0,
        maintenance_needed=False,
        anomaly_score=0.1,
        confidence=0.95,
        model_version="1.0",
    )
    session.add(pred)
    session.commit()
    result = session.query(VehiclePrediction).filter_by(vehicle_id="v001").first()
    assert result is not None
    assert result.speed_kmh == 60.0
    session.close()


def test_insert_drift_report(in_memory_db):
    Session = sessionmaker(bind=in_memory_db)
    session = Session()
    report = DriftReport(
        feature_name="speed_kmh",
        ks_statistic=0.3,
        p_value=0.02,
        drift_detected=True,
        sample_size=100,
    )
    session.add(report)
    session.commit()
    result = session.query(DriftReport).filter_by(feature_name="speed_kmh").first()
    assert result is not None
    assert result.drift_detected is True
    session.close()


def test_vehicle_prediction_maintenance_flag(in_memory_db):
    Session = sessionmaker(bind=in_memory_db)
    session = Session()
    pred = VehiclePrediction(
        vehicle_id="v002",
        speed_kmh=0.0,
        engine_temp_c=120.0,
        oil_pressure_bar=0.5,
        fuel_consumption_lph=0.0,
        battery_voltage_v=11.0,
        brake_pad_mm=2.0,
        tire_pressure_bar=1.5,
        rpm=0.0,
        coolant_temp_c=110.0,
        transmission_temp_c=90.0,
        maintenance_needed=True,
        anomaly_score=0.95,
        confidence=0.8,
        model_version="1.0",
    )
    session.add(pred)
    session.commit()
    result = session.query(VehiclePrediction).filter_by(vehicle_id="v002").first()
    assert result.maintenance_needed is True
    session.close()
