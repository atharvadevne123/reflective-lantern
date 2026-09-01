"""Shared pytest fixtures for Threat-Lens test suite."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app

TEST_DATABASE_URL = "sqlite:///./test_threat_lens.db"

_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=_engine)
    yield
    Base.metadata.drop_all(bind=_engine)


@pytest.fixture()
def db_session():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


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


@pytest.fixture()
def sample_normal_flow():
    return {
        "duration": 5.0,
        "src_bytes": 1000,
        "dst_bytes": 2000,
        "protocol_type": "tcp",
        "service": "http",
        "flag": "SF",
        "logged_in": 1,
        "count": 3,
        "serror_rate": 0.0,
        "rerror_rate": 0.0,
        "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0,
        "dst_host_count": 5,
        "dst_host_srv_count": 5,
    }


@pytest.fixture()
def sample_dos_flow():
    return {
        "duration": 0.0,
        "src_bytes": 0,
        "dst_bytes": 0,
        "protocol_type": "tcp",
        "service": "http",
        "flag": "S0",
        "logged_in": 0,
        "count": 511,
        "serror_rate": 1.0,
        "rerror_rate": 0.0,
        "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0,
        "dst_host_count": 255,
        "dst_host_srv_count": 255,
    }


@pytest.fixture()
def trained_model():
    from app.features import generate_synthetic_dataset
    from app.model import train_model

    X, y = generate_synthetic_dataset(n_samples=500, seed=0)
    pipe, metrics = train_model(X, y)
    return pipe, metrics
