"""Pytest fixtures and test database setup for Cyber-Guard."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app


@pytest.fixture(scope="session")
def test_engine():
    engine = create_engine("sqlite:///./test_cyber_guard.db", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    import os
    if os.path.exists("test_cyber_guard.db"):
        os.remove("test_cyber_guard.db")


@pytest.fixture
def db_session(test_engine):
    Session = sessionmaker(bind=test_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
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


@pytest.fixture
def sample_request_payload():
    return {
        "src_bytes": 491,
        "dst_bytes": 0,
        "duration": 0.0,
        "protocol_type": "tcp",
        "service": "http",
        "flag": "SF",
    }


@pytest.fixture
def sample_dataframe():
    import pandas as pd
    return pd.DataFrame([{
        "src_bytes": 100.0,
        "dst_bytes": 200.0,
        "duration": 1.0,
        "protocol_type": "tcp",
        "service": "http",
        "flag": "SF",
    }])
