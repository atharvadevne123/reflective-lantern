"""Pytest fixtures and test database configuration for Quake-Net tests."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_quake_net.db")
os.environ.setdefault("MODEL_PATH", "/tmp/test_quake_model.joblib")
os.environ.setdefault("METRICS_PATH", "/tmp/test_quake_metrics.json")

from app.database import Base, get_db
from app.features import make_synthetic_dataset

TEST_ENGINE = create_engine(
    "sqlite:///./test_quake_net.db",
    connect_args={"check_same_thread": False},
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=TEST_ENGINE)


@pytest.fixture(scope="session", autouse=True)
def create_test_tables():
    Base.metadata.create_all(bind=TEST_ENGINE)
    yield
    Base.metadata.drop_all(bind=TEST_ENGINE)
    Path("./test_quake_net.db").unlink(missing_ok=True)


@pytest.fixture
def db_session():
    connection = TEST_ENGINE.connect()
    transaction = connection.begin()
    session = TestSessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def trained_model():
    from app.model import train_model

    df = make_synthetic_dataset(n_samples=300, seed=0)
    pipeline, metrics = train_model(df=df)
    return pipeline, metrics


@pytest.fixture
def sample_features() -> dict:
    return {
        "latitude": 37.5,
        "longitude": -122.0,
        "depth_km": 10.0,
        "station_count": 15,
        "p_wave_amplitude": 3.5,
        "s_wave_amplitude": 6.2,
        "epicentral_distance_km": 80.0,
        "fault_type": "strike_slip",
    }


@pytest.fixture
def small_dataset() -> pd.DataFrame:
    return make_synthetic_dataset(n_samples=200, seed=42)


@pytest.fixture
def mock_model():
    """A fast mock pipeline for API tests that avoids full ML training."""
    mock = MagicMock()
    mock.predict.return_value = np.array([5.2])
    return mock


@pytest.fixture
def app_client(trained_model, db_session):
    from app.main import _model_cache, app

    pipeline, _ = trained_model
    _model_cache["pipeline"] = pipeline

    def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


VALID_EVENT = {
    "latitude": 35.6,
    "longitude": 139.7,
    "depth_km": 20.0,
    "station_count": 12,
    "p_wave_amplitude": 4.1,
    "s_wave_amplitude": 7.8,
    "epicentral_distance_km": 100.0,
    "fault_type": "reverse",
}
