"""Pytest fixtures and test database setup."""

from __future__ import annotations

import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_ticket_scout.db")
os.environ.setdefault("MODEL_DIR", "./test_models")

from app.database import Base, get_db  # noqa: E402


@pytest.fixture(scope="session")
def test_engine():
    engine = create_engine(
        "sqlite:///./test_ticket_scout.db",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    import os as _os
    for path in ["./test_ticket_scout.db"]:
        if _os.path.exists(path):
            _os.remove(path)


@pytest.fixture
def db_session(test_engine):
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSession()
    yield session
    session.rollback()
    session.close()


@pytest.fixture(scope="session")
def trained_models():
    """Train models once per session and return the three pipelines."""
    from app.model import generate_synthetic_data, load_models, train_models

    os.makedirs("./test_models", exist_ok=True)
    os.environ["MODEL_DIR"] = "./test_models"

    df = generate_synthetic_data(n_samples=400)
    train_models(df)
    cat_pipe, breach_pipe, res_pipe = load_models()
    return cat_pipe, breach_pipe, res_pipe


@pytest.fixture
def sample_ticket_df():
    """A minimal single-row ticket DataFrame."""
    return pd.DataFrame([{
        "subject": "VPN not connecting after the update",
        "body": "My VPN drops every 10 minutes. Blocking all remote work.",
        "priority": "high",
        "created_hour": 9,
        "created_dow": 1,
        "org_size": 500,
        "lag_resolution_hours": 8.0,
        "rolling_breach_rate": 0.15,
    }])


@pytest.fixture
def sample_batch_df():
    """A small batch of tickets for batch prediction tests."""
    return pd.DataFrame([
        {
            "subject": "Password reset needed",
            "body": "Account locked out. Please reset my password urgently.",
            "priority": "critical",
            "created_hour": 8,
            "created_dow": 0,
            "org_size": 200,
            "lag_resolution_hours": 2.0,
            "rolling_breach_rate": 0.4,
        },
        {
            "subject": "Outlook not syncing",
            "body": "Emails stopped arriving since yesterday. Low priority.",
            "priority": "low",
            "created_hour": 15,
            "created_dow": 3,
            "org_size": 1000,
            "lag_resolution_hours": 24.0,
            "rolling_breach_rate": 0.05,
        },
        {
            "subject": "Laptop screen flickering",
            "body": "Screen flickers intermittently when unplugged from charger.",
            "priority": "medium",
            "created_hour": 10,
            "created_dow": 2,
            "org_size": 750,
            "lag_resolution_hours": 8.0,
            "rolling_breach_rate": 0.1,
        },
    ])


@pytest.fixture
def client(db_session):
    """Test client with overridden DB dependency."""
    from app.main import app

    def _override_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
