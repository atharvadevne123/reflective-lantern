"""Pytest fixtures and test database setup."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app

TEST_DB_URL = "sqlite:///./test_watt_guard.db"

test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def create_test_db() -> None:
    """Create test DB tables once per session and drop them after."""
    Base.metadata.create_all(bind=test_engine)
    yield  # type: ignore[misc]
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session():
    """Yield a fresh SQLAlchemy session for each test."""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session):
    """FastAPI TestClient wired to the in-memory test database."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """300-row synthetic energy DataFrame for model tests."""
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame(
        {
            "hour": rng.integers(0, 24, n),
            "day_of_week": rng.integers(0, 7, n),
            "month": rng.integers(1, 13, n),
            "temperature_c": rng.uniform(5, 35, n),
            "humidity_pct": rng.uniform(30, 80, n),
            "occupancy": rng.integers(0, 150, n),
            "hvac_state": rng.integers(0, 2, n),
            "consumption_kwh": 8 + rng.normal(0, 3, n).clip(-7),
        }
    )


@pytest.fixture
def sample_y(sample_df: pd.DataFrame) -> pd.Series:
    return sample_df["consumption_kwh"].copy()


@pytest.fixture
def trained_model(sample_df, sample_y):
    from app.model import train_model

    bundle, metrics = train_model(sample_df, sample_y)
    return bundle, metrics


@pytest.fixture
def trained_anomaly_model(sample_df):
    from app.model import train_anomaly_model

    return train_anomaly_model(sample_df)


@pytest.fixture
def energy_payload():
    return {
        "building_id": "bldg-001",
        "timestamp": "2025-06-01T14:00:00",
        "hour": 14,
        "day_of_week": 1,
        "month": 6,
        "temperature_c": 28.5,
        "humidity_pct": 60.0,
        "occupancy": 50,
        "hvac_state": 1,
        "consumption_kwh": 12.0,
    }
