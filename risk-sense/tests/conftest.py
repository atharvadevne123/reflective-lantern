"""Pytest fixtures and test database for Risk-Sense."""

from __future__ import annotations

import sys
import os

# Add project root to path so app imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app
from app.features import build_pipeline
from app.model import generate_synthetic_data, train


TEST_DATABASE_URL = "sqlite:///./test_risk_sense.db"

test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def create_test_tables() -> None:
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)
    if os.path.exists("./test_risk_sense.db"):
        os.remove("./test_risk_sense.db")


@pytest.fixture()
def db_session():
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture()
def client(db_session) -> TestClient:
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
def trained_model_and_pipeline():
    X, y = generate_synthetic_data(n_samples=500, seed=42)
    pipeline = build_pipeline()
    model, metrics = train(X, y, pipeline)
    return model, pipeline, metrics


@pytest.fixture()
def sample_application() -> dict:
    return {
        "applicant_id": "test-applicant-001",
        "debt_to_income": 0.35,
        "payment_history_score": 0.85,
        "credit_utilization": 0.30,
        "account_age_months": 48,
        "num_open_accounts": 4,
        "num_derogatory_marks": 0,
        "annual_income": 75_000.0,
    }


@pytest.fixture()
def high_risk_application() -> dict:
    return {
        "applicant_id": "test-high-risk-001",
        "debt_to_income": 0.75,
        "payment_history_score": 0.15,
        "credit_utilization": 0.92,
        "account_age_months": 6,
        "num_open_accounts": 8,
        "num_derogatory_marks": 4,
        "annual_income": 20_000.0,
    }
