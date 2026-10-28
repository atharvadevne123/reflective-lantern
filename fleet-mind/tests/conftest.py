"""Shared pytest fixtures for Fleet-Mind tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure the fleet-mind package root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_fleet_mind.db")
os.environ.setdefault("MODEL_PATH", "/tmp/test_fleet_model.joblib")
os.environ.setdefault("METRICS_PATH", "/tmp/test_fleet_metrics.json")

from app.database import Base, get_db
from app.main import app

TEST_DB_URL = "sqlite:///./test_fleet_mind.db"


@pytest.fixture(scope="session")
def engine():
    """Create a test database engine."""
    eng = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    yield eng
    Base.metadata.drop_all(bind=eng)
    Path("test_fleet_mind.db").unlink(missing_ok=True)


@pytest.fixture(scope="session")
def db_session(engine):
    """Yield a session connected to the test database."""
    factory = sessionmaker(bind=engine)
    session = factory()
    yield session
    session.close()


@pytest.fixture(scope="session")
def client(engine):
    """Return a FastAPI test client with an overridden DB session."""
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def sample_reading() -> dict:
    """Return a normal-range telemetry reading."""
    return {
        "vehicle_id": "TEST-001",
        "speed_kmh": 80.0,
        "engine_temp_c": 90.0,
        "oil_pressure_bar": 3.5,
        "fuel_consumption_lph": 8.0,
        "battery_voltage_v": 12.8,
        "brake_pad_mm": 7.0,
        "tire_pressure_bar": 2.3,
        "rpm": 2500.0,
        "coolant_temp_c": 85.0,
        "transmission_temp_c": 75.0,
    }


@pytest.fixture
def anomalous_reading() -> dict:
    """Return an anomalous telemetry reading that should trigger maintenance."""
    return {
        "vehicle_id": "TEST-999",
        "speed_kmh": 100.0,
        "engine_temp_c": 125.0,
        "oil_pressure_bar": 0.8,
        "fuel_consumption_lph": 20.0,
        "battery_voltage_v": 10.5,
        "brake_pad_mm": 1.5,
        "tire_pressure_bar": 1.2,
        "rpm": 4800.0,
        "coolant_temp_c": 108.0,
        "transmission_temp_c": 135.0,
    }
