"""Pytest fixtures for the Traffic-Pulse test suite."""

from __future__ import annotations

import os
from collections.abc import Generator
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_tp.db")
os.environ.setdefault("MODEL_PATH", "/tmp/test_model.joblib")
os.environ.setdefault("METRICS_PATH", "/tmp/test_metrics.json")

from app.database import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402

_TEST_DB = "sqlite:///./test_tp.db"
_test_engine = create_engine(_TEST_DB, connect_args={"check_same_thread": False})
_TestSession = sessionmaker(autocommit=False, autoflush=False, bind=_test_engine)


@pytest.fixture(scope="session", autouse=True)
def create_tables() -> Generator[None, None, None]:
    Base.metadata.create_all(bind=_test_engine)
    yield
    Base.metadata.drop_all(bind=_test_engine)


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    session = _TestSession()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def mock_models() -> dict:
    pipe = MagicMock()
    pipe.predict_proba.return_value = np.array([[0.60, 0.28, 0.09, 0.03]])
    return {"xgb": pipe, "lgbm": pipe}


@pytest.fixture
def client(mock_models: dict) -> Generator[TestClient, None, None]:
    def _override_db() -> Generator[Session, None, None]:
        s = _TestSession()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _override_db

    from app import main as m

    m._models.update(mock_models)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    m._models.clear()


@pytest.fixture
def sample_payload() -> dict:
    return {
        "route_id": "R001",
        "hour": 8,
        "day_of_week": 1,
        "month": 6,
        "vehicle_count": 1500,
        "avg_speed_kmh": 35.0,
        "road_type": "arterial",
        "incident_count": 2,
        "temperature_celsius": 22.0,
        "is_raining": 0,
    }


@pytest.fixture
def reference_distribution() -> list[float]:
    return np.random.default_rng(0).normal(50, 10, 100).tolist()


@pytest.fixture
def drifted_distribution() -> list[float]:
    return np.random.default_rng(1).normal(80, 10, 100).tolist()
