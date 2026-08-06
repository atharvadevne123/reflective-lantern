"""Pytest fixtures for Temporal-Pulse test suite."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Generator

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Use in-memory SQLite for all tests
TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="session")
def engine():
    """Create a test SQLAlchemy engine."""
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from sqlalchemy.pool import StaticPool

    from app.database import Base

    eng = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture()
def db_session(engine):
    """Return a transactional database session that rolls back after each test."""
    TestSession = sessionmaker(bind=engine)
    session = TestSession()
    yield session
    session.rollback()
    session.close()


@pytest.fixture(scope="session")
def client(engine) -> Generator:
    """Return a TestClient with the DB patched to use the in-memory engine."""
    import os

    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    os.environ["RATE_LIMIT_ENABLED"] = "false"

    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from app.database import Base, get_db
    from app.main import app

    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


@pytest.fixture()
def sample_readings() -> list[dict]:
    """Generate a batch of synthetic multivariate sensor readings."""
    base_time = datetime(2026, 1, 1, 0, 0, 0)
    readings = []
    rng = np.random.default_rng(42)
    for i in range(50):
        ts = base_time + timedelta(minutes=i)
        readings.append(
            {
                "sensor_id": "sensor-001",
                "timestamp": ts.isoformat(),
                "values": {
                    "temperature": float(rng.normal(22.0, 0.5)),
                    "humidity": float(rng.normal(60.0, 2.0)),
                    "pressure": float(rng.normal(1013.0, 3.0)),
                    "vibration": float(rng.normal(0.1, 0.02)),
                },
            }
        )
    return readings


@pytest.fixture()
def anomalous_reading() -> dict:
    """Return a single reading with extreme values that should be flagged."""
    return {
        "sensor_id": "sensor-anomaly",
        "timestamp": "2026-01-01T12:00:00",
        "values": {
            "temperature": 999.0,
            "humidity": -50.0,
            "pressure": 500.0,
            "vibration": 100.0,
        },
    }


@pytest.fixture()
def feature_matrix(sample_readings) -> tuple:
    """Build a feature matrix from sample readings."""
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from app.features import build_feature_matrix

    raw = [
        {"timestamp": r["timestamp"], "sensor_id": r["sensor_id"], **r["values"]}
        for r in sample_readings
    ]
    return build_feature_matrix(raw)


@pytest.fixture()
def trained_models(feature_matrix):
    """Train anomaly detector and forecaster on sample data and return both."""
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    import numpy as np

    from app.model import train_anomaly_detector, train_forecaster

    df, feature_cols = feature_matrix
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = X[:, 0]
    if_model, scaler = train_anomaly_detector(X, contamination=0.05)
    rf_model, metrics = train_forecaster(X, y)
    return {
        "if_model": if_model,
        "scaler": scaler,
        "rf_model": rf_model,
        "metrics": metrics,
        "X": X,
        "feature_cols": feature_cols,
    }
