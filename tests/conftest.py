"""Shared pytest fixtures for Volt-Cast tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app
from app.model import generate_synthetic_training_data

TEST_DB_URL = "sqlite:///./test_volt_cast.db"

test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session")
def db_engine():
    Base.metadata.create_all(bind=test_engine)
    yield test_engine
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture()
def db_session(db_engine):
    connection = db_engine.connect()
    transaction = connection.begin()
    session = TestSessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture()
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def synthetic_data():
    X, y = generate_synthetic_training_data(n_samples=300)
    return X, y


@pytest.fixture()
def sample_predict_payload():
    return {
        "hour": 14,
        "day_of_week": 2,
        "month": 7,
        "is_weekend": False,
        "temperature_c": 28.5,
        "humidity_pct": 65.0,
        "historical_loads": [3500.0, 3600.0, 3800.0, 4000.0, 4200.0],
        "region": "northeast",
    }
