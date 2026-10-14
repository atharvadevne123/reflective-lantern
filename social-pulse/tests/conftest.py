"""Shared pytest fixtures for Social-Pulse test suite."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app
from app.model import generate_synthetic_data

TEST_DATABASE_URL = "sqlite:///./test_social_pulse.db"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db() -> None:
    """Create all tables once for the test session."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session():
    """Yield a fresh database session; roll back after each test."""
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(db_session) -> TestClient:
    """Return a TestClient that uses the in-memory test DB."""

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
def synthetic_samples() -> tuple[list[dict], list[int]]:
    """Return 100 synthetic training samples."""
    return generate_synthetic_data(n=100)


@pytest.fixture
def sample_post() -> dict:
    """Return a single valid post feature dict."""
    return {
        "text": "Breaking: our new product launch is live! #innovation #tech @everyone",
        "author_followers": 50000,
        "has_media": True,
        "has_link": True,
        "hashtag_count": 2,
        "mention_count": 1,
        "hour_of_day": 18,
        "day_of_week": 2,
    }


@pytest.fixture
def viral_post() -> dict:
    """Return a post that should score high on virality."""
    return {
        "text": "#BREAKING Just went viral! Check this out!! @media @news #trending #viral #omg",
        "author_followers": 1_000_000,
        "has_media": True,
        "has_link": True,
        "hashtag_count": 5,
        "mention_count": 2,
        "hour_of_day": 19,
        "day_of_week": 5,
    }


@pytest.fixture
def low_engagement_post() -> dict:
    """Return a post that should score low on virality."""
    return {
        "text": "just had lunch",
        "author_followers": 50,
        "has_media": False,
        "has_link": False,
        "hashtag_count": 0,
        "mention_count": 0,
        "hour_of_day": 14,
        "day_of_week": 1,
    }
