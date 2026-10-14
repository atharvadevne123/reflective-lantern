"""Pytest fixtures for Trade-Sage tests."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure app is importable from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy.pool import StaticPool

TEST_DB_URL = "sqlite://"  # in-memory, shared via StaticPool


@pytest.fixture(scope="session")
def db_engine():
    from app.database import Base

    engine = create_engine(
        TEST_DB_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session(db_engine):
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()


@pytest.fixture()
def sample_ohlcv_df() -> pd.DataFrame:
    """100-row synthetic OHLCV dataframe."""
    rng = np.random.default_rng(0)
    n = 100
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


@pytest.fixture()
def sample_features(sample_ohlcv_df) -> np.ndarray:
    from app.features import engineer_features

    return engineer_features(sample_ohlcv_df)


@pytest.fixture()
def trained_model(sample_features):
    """Quickly trained model for tests."""
    import numpy as np

    os.environ["MODEL_PATH"] = "/tmp/test_trade_sage_model.joblib"
    os.environ["METRICS_PATH"] = "/tmp/test_trade_sage_metrics.json"
    from app.model import train_model

    X = np.tile(sample_features, (5, 1))
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, len(X))
    pipe, metrics = train_model(X, y)
    return pipe, metrics


@pytest.fixture()
def ohlcv_payload(sample_ohlcv_df) -> list[dict]:
    return sample_ohlcv_df.to_dict(orient="records")


@pytest.fixture()
def fastapi_client(db_session):
    """TestClient with overridden DB dependency."""
    import os

    os.environ["MODEL_PATH"] = "/tmp/test_trade_sage_model.joblib"
    os.environ["METRICS_PATH"] = "/tmp/test_trade_sage_metrics.json"
    from fastapi.testclient import TestClient

    from app.database import get_db
    from app.main import app

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
