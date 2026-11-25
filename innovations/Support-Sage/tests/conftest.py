"""Shared pytest fixtures for Support-Sage tests."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app

TEST_DATABASE_URL = "sqlite:///./test_support_sage.db"


@pytest.fixture(scope="session")
def test_engine():
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def db_session(test_engine):
    Session = sessionmaker(bind=test_engine)
    session = Session()
    try:
        yield session
    finally:
        session.rollback()
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
def sample_ticket():
    return {
        "ticket_id": "TEST-001",
        "subject": "Cannot connect to VPN from home office",
        "description": "Since this morning I am unable to establish VPN connection. Getting error code 800. Already restarted laptop twice.",
    }


@pytest.fixture()
def critical_ticket():
    return {
        "ticket_id": "CRIT-001",
        "subject": "CRITICAL: Production server down - all users affected",
        "description": "Production server is completely down. Emergency. All 500 users cannot access the system. Outage started at 09:00 UTC.",
    }


@pytest.fixture()
def sample_df():
    from app.features import generate_synthetic_data
    return generate_synthetic_data(n_samples=200)
