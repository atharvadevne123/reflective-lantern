"""Shared pytest fixtures for Logistics-Flow tests."""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """Return a small synthetic DataFrame for feature tests."""
    return pd.DataFrame(
        {
            "carrier": ["DHL", "FedEx", "UPS", "USPS", "Amazon"] * 4,
            "distance_km": [10.0, 50.0, 200.0, 500.0, 30.0] * 4,
            "weight_kg": [1.0, 5.0, 15.0, 30.0, 0.5] * 4,
            "route_type": ["urban", "suburban", "rural", "highway", "urban"] * 4,
            "hour_of_day": [8, 12, 17, 22, 6] * 4,
            "day_of_week": [0, 1, 5, 6, 3] * 4,
            "delivery_minutes": [45.0, 120.0, 400.0, 900.0, 60.0] * 4,
        }
    )


@pytest.fixture(scope="session")
def test_engine():
    """Shared in-memory SQLite engine for tests.

    StaticPool keeps every session on the same connection, so the schema
    created here is visible to the TestClient's worker thread too.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture()
def db_session(test_engine):
    """Yield an isolated test DB session."""
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestSession()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture()
def client(test_engine):
    """TestClient that overrides the DB dependency."""
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    def _override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def predict_payload() -> dict:
    return {
        "carrier": "DHL",
        "distance_km": 42.5,
        "weight_kg": 3.2,
        "route_type": "urban",
        "hour_of_day": 14,
        "day_of_week": 2,
    }
