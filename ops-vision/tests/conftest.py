"""Pytest fixtures for Ops-Vision test suite."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.features import build_feature_pipeline
from app.main import app
from app.model import generate_synthetic_data

TEST_DATABASE_URL = "sqlite:///./test_ops_vision.db"

test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create all tables in the SQLite test database."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session():
    """Yield a test database session, rolling back after each test."""
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def client(db_session):
    """Return a TestClient with the DB dependency overridden."""

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def sample_metrics() -> dict:
    """Return a valid metrics payload dict for API tests."""
    return {
        "service_name": "payments-api",
        "cpu_usage_pct": 85.0,
        "memory_usage_pct": 88.0,
        "error_rate_per_min": 62.0,
        "latency_p99_ms": 1450.0,
        "request_rate_per_sec": 45.0,
        "disk_io_util_pct": 80.0,
    }


@pytest.fixture
def normal_metrics() -> dict:
    """Return a normal (non-incident) metrics payload."""
    return {
        "service_name": "user-service",
        "cpu_usage_pct": 35.0,
        "memory_usage_pct": 42.0,
        "error_rate_per_min": 1.5,
        "latency_p99_ms": 120.0,
        "request_rate_per_sec": 250.0,
        "disk_io_util_pct": 20.0,
    }


@pytest.fixture
def synthetic_dataframe() -> pd.DataFrame:
    """Return a synthetic feature DataFrame for pipeline tests."""
    df, _ = generate_synthetic_data(n_samples=200)
    return df


@pytest.fixture
def synthetic_labels(synthetic_dataframe) -> np.ndarray:
    """Return labels matching synthetic_dataframe."""
    _, labels = generate_synthetic_data(n_samples=200)
    return labels.values


@pytest.fixture
def fitted_pipeline(synthetic_dataframe):
    """Return a pipeline fitted on synthetic data."""
    pipeline = build_feature_pipeline()
    pipeline.fit(synthetic_dataframe)
    return pipeline


@pytest.fixture
def transformed_X(fitted_pipeline, synthetic_dataframe) -> np.ndarray:
    """Return the transformed feature matrix."""
    return fitted_pipeline.transform(synthetic_dataframe)


@pytest.fixture
def trained_model(transformed_X, synthetic_labels):
    """Return a trained VotingClassifier on synthetic data."""
    from app.model import train

    model, _ = train(transformed_X, synthetic_labels, cv_folds=2)
    return model
