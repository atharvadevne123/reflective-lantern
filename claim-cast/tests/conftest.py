"""Pytest fixtures and test database setup."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app

TEST_DB_URL = "sqlite:///./test_claim_cast.db"
test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session():
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def sample_claim() -> dict:
    return {
        "claim_id": "CLM-TEST-001",
        "claim_age_days": 15.0,
        "policy_limit": 100_000.0,
        "claimed_amount": 25_000.0,
        "num_prior_claims": 1.0,
        "policy_tenure_years": 3.0,
        "geo_risk_tier": 2.5,
        "days_since_last_claim": 400.0,
        "policy_type": "auto",
    }


@pytest.fixture
def high_severity_claim() -> dict:
    return {
        "claim_id": "CLM-HIGH-001",
        "claim_age_days": 2.0,
        "policy_limit": 100_000.0,
        "claimed_amount": 90_000.0,
        "num_prior_claims": 5.0,
        "policy_tenure_years": 1.0,
        "geo_risk_tier": 5.0,
        "days_since_last_claim": 30.0,
        "policy_type": "commercial",
    }
