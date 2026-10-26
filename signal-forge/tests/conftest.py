"""Pytest fixtures and test database configuration for Signal-Forge."""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.database import Base, get_db

TEST_DATABASE_URL = "sqlite:///./test_signal_forge.db"


@pytest.fixture(scope="session")
def test_engine():
    """Create an in-memory SQLite engine for tests."""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(test_engine):
    """Provide a transactional test database session."""
    TestingSessionLocal = sessionmaker(bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def client(db_session):
    """FastAPI test client with overridden DB dependency."""
    from app.main import app

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def sample_ohlcv_df():
    """Generate a synthetic OHLCV DataFrame with 70 rows (enough for 63-day beta window)."""
    rng = np.random.default_rng(42)
    n = 70
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        "market_return": rng.normal(0.0005, 0.01, n),
    })


@pytest.fixture
def sample_feature_array():
    """Return a synthetic (50, 5) feature array for model tests."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((50, 5)).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Return synthetic regime labels for 50 samples."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 4, 50)


@pytest.fixture
def predict_payload():
    """Minimal valid /predict request payload."""
    return {"ticker": "AAPL", "close": 195.0, "volume": 80_000_000.0}
