"""Pytest fixtures and test database setup."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app

TEST_DATABASE_URL = "sqlite:///./test_energy_seer.db"

engine_test = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine_test)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True, scope="session")
def setup_test_db():
    Base.metadata.create_all(bind=engine_test)
    yield
    Base.metadata.drop_all(bind=engine_test)


@pytest.fixture(scope="session")
def client():
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def sample_reading() -> dict:
    return {
        "meter_id": "meter_001",
        "consumption_kwh": 4.5,
        "temperature_c": 22.0,
        "humidity_pct": 55.0,
        "hour_of_day": 14,
        "day_of_week": 2,
        "is_holiday": False,
        "building_type": "residential",
    }


@pytest.fixture
def sample_readings(sample_reading) -> list[dict]:
    return [
        sample_reading,
        {**sample_reading, "meter_id": "meter_002", "consumption_kwh": 8.2, "building_type": "commercial"},
        {**sample_reading, "meter_id": "meter_003", "consumption_kwh": 2.1, "hour_of_day": 3},
    ]


@pytest.fixture
def high_consumption_reading(sample_reading) -> dict:
    return {**sample_reading, "consumption_kwh": 95.0, "temperature_c": 38.0}


@pytest.fixture
def low_consumption_reading(sample_reading) -> dict:
    return {**sample_reading, "consumption_kwh": 0.01, "hour_of_day": 4}
