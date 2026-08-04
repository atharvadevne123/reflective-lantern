"""Pytest fixtures for Forge-Guard test suite."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app

TEST_DATABASE_URL = "sqlite:///./test_forge_guard.db"

engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create all tables in the test SQLite database once per session."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session():
    """Return a fresh database session per test, rolling back after."""
    connection = engine.connect()
    transaction = connection.begin()
    session = TestSessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="module")
def client():
    """Return a TestClient with the test DB injected."""
    app.dependency_overrides[get_db] = override_get_db
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def sample_sensor_payload() -> dict:
    return {
        "temperature": 78.5,
        "pressure": 52.0,
        "vibration": 2.1,
        "cycle_time": 28.0,
        "tool_wear": 15.0,
        "power_consumption": 98.0,
        "humidity": 45.0,
    }


@pytest.fixture
def high_risk_payload() -> dict:
    """Sensor reading expected to produce a high defect probability."""
    return {
        "temperature": 92.0,
        "pressure": 61.0,
        "vibration": 7.5,
        "cycle_time": 28.0,
        "tool_wear": 55.0,
        "power_consumption": 135.0,
        "humidity": 65.0,
    }


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    from app.features import generate_synthetic_data

    return generate_synthetic_data(n_samples=300, seed=0)


@pytest.fixture
def feature_arrays(synthetic_df: pd.DataFrame):
    from app.features import build_feature_pipeline

    feat_cols = [c for c in synthetic_df.columns if c != "defect"]
    pipe = build_feature_pipeline()
    X = pipe.fit_transform(synthetic_df[feat_cols])
    y = synthetic_df["defect"].values
    return X, y


@pytest.fixture
def low_risk_payload() -> dict:
    """Sensor reading expected to produce a low defect probability."""
    return {
        "temperature": 70.0,
        "pressure": 48.0,
        "vibration": 1.0,
        "cycle_time": 32.0,
        "tool_wear": 5.0,
        "power_consumption": 85.0,
        "humidity": 40.0,
    }


@pytest.fixture
def boundary_payload() -> dict:
    """Sensor reading at the boundary of valid ranges."""
    return {
        "temperature": -50.0,
        "pressure": 0.0,
        "vibration": 0.0,
        "cycle_time": 0.1,
        "tool_wear": 0.0,
        "power_consumption": 0.0,
        "humidity": 0.0,
    }


@pytest.fixture
def large_synthetic_df() -> pd.DataFrame:
    from app.features import generate_synthetic_data

    return generate_synthetic_data(n_samples=1000, seed=99)
