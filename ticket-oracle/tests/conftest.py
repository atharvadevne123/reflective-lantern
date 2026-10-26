"""Shared pytest fixtures for Ticket-Oracle test suite."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app


@pytest.fixture(scope="session")
def test_engine():
    """In-memory SQLite engine for the full test session."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session(test_engine):
    """Transactional DB session that rolls back after each test."""
    connection = test_engine.connect()
    transaction = connection.begin()
    factory = sessionmaker(autocommit=False, autoflush=False, bind=connection)
    session = factory()
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture()
def client(db_session):
    """TestClient with the DB dependency overridden."""
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def sample_ticket_payload() -> dict:
    """A valid ticket payload for prediction tests."""
    return {
        "ticket_id": "T-99999",
        "description": "VPN connectivity failure affecting Engineering remote workers after Windows update",
        "department": "Engineering",
        "incident_type": "Network",
        "channel": "email",
        "customer_tier": "Gold",
        "product_area": "Infrastructure",
        "open_tickets_count": 45,
        "agent_load": 8,
        "hour_of_day": 14,
        "day_of_week": 1,
        "day_of_month": 15,
        "ticket_age_minutes": 30.0,
        "same_user_last_7d": 2,
    }


@pytest.fixture()
def sample_p4_payload() -> dict:
    """A low-priority ticket payload."""
    return {
        "ticket_id": "T-00001",
        "description": "Request to provision cloud storage bucket for Marketing project",
        "department": "Marketing",
        "incident_type": "Service Request",
        "channel": "portal",
        "customer_tier": "Bronze",
        "product_area": "Unknown",
        "open_tickets_count": 5,
        "agent_load": 2,
        "hour_of_day": 10,
        "day_of_week": 2,
        "day_of_month": 10,
        "ticket_age_minutes": 5.0,
        "same_user_last_7d": 0,
    }
