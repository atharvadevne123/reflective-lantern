"""Shared pytest fixtures for Logistics-Flow tests."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.monitoring import reset_anomaly_flags_buffer


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """Return a small synthetic DataFrame for feature tests."""
    return pd.DataFrame(
        {
            "carrier": ["DHL", "FedEx", "UPS", "USPS", "Amazon"] * 4,
            "distance_km": [10.0, 50.0, 200.0, 500.0, 30.0] * 4,
            "weight_kg": [1.0, 5.0, 15.0, 30.0, 0.5] * 4,
            "route_type": ["urban", "suburban", "rural", "highway", "urban"] * 4,
            "hour_of_day": [8, 12, 17, 22, 6] * 4,
            "day_of_week": [0, 1, 5, 6, 3] * 4,
            "delivery_minutes": [45.0, 120.0, 400.0, 900.0, 60.0] * 4,
        }
    )


@pytest.fixture(scope="session")
def test_engine():
    """Shared in-memory SQLite engine for tests.

    StaticPool keeps every session on the same connection, so the schema
    created here is visible to the TestClient's worker thread too.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture()
def db_session(test_engine):
    """Yield a per-test DB session that cleans up after itself.

    Uses a connection-level SAVEPOINT so that each test's commits are
    visible within the test but rolled back at teardown, keeping the
    shared in-memory database clean.
    """
    connection = test_engine.connect()
    transaction = connection.begin()
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=connection)
    session = TestSession()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture()
def client(test_engine):
    """TestClient that overrides the DB dependency."""
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    def _override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _reset_monitoring_globals() -> None:
    """Reset in-memory monitoring buffers before every test."""
    reset_anomaly_flags_buffer()


@pytest.fixture()
def predict_payload() -> dict:
    return {
        "carrier": "DHL",
        "distance_km": 42.5,
        "weight_kg": 3.2,
        "route_type": "urban",
        "hour_of_day": 14,
        "day_of_week": 2,
    }


@pytest.fixture()
def history_dir(tmp_path: Path) -> Path:
    """History directory with SampleRepo (2 entries) and AnotherRepo (1 entry)."""
    h = tmp_path / "history"
    h.mkdir()
    (h / "SampleRepo.json").write_text(json.dumps([
        {"date": "2026-06-10", "commits": 60, "mode": "improvement",
         "improvements": ["fix A", "fix B"], "tests_passed": True, "email_status": "sent"},
        {"date": "2026-06-15", "commits": 60, "mode": "improvement",
         "improvements": ["fix C", "fix D"], "tests_passed": True, "email_status": "sent"},
    ]))
    (h / "AnotherRepo.json").write_text(json.dumps([
        {"date": "2026-06-12", "commits": 45, "mode": "improvement",
         "improvements": ["fix E"], "tests_passed": True, "email_status": ""},
    ]))
    return h


@pytest.fixture()
def single_entry_history_dir(tmp_path: Path) -> Path:
    """History directory with one file having a last_run fallback date."""
    h = tmp_path / "history"
    h.mkdir()
    (h / "MyRepo.json").write_text(json.dumps([
        {"last_run": "2026-06-20", "commits": 60, "mode": "improvement",
         "tests_passed": True, "email_status": ""},
    ]))
    return h


@pytest.fixture()
def invalid_history_dir(tmp_path: Path) -> Path:
    """History directory with an invalid JSON file."""
    h = tmp_path / "history"
    h.mkdir()
    (h / "Bad.json").write_text("not valid json {{{")
    return h


@pytest.fixture()
def multi_repo_history_dir(tmp_path: Path) -> Path:
    """History directory with Alpha, Beta, Gamma repos for multi-repo tests."""
    h = tmp_path / "history"
    h.mkdir()
    (h / "Alpha.json").write_text(json.dumps([
        {"date": "2026-07-01", "commits": 60, "mode": "IMPROVEMENT",
         "improvements": [], "tests_passed": True, "email_status": ""},
        {"date": "2026-07-05", "commits": 60, "mode": "improvement",
         "improvements": [], "tests_passed": True, "email_status": ""},
    ]))
    (h / "Beta.json").write_text(json.dumps([
        {"date": "2026-07-03", "commits": 120, "mode": "Innovation",
         "improvements": [], "tests_passed": True, "email_status": ""},
    ]))
    (h / "Gamma.json").write_text(json.dumps([
        {"date": "2026-07-06", "commits": 60, "mode": "INNOVATION",
         "improvements": [], "tests_passed": True, "email_status": ""},
    ]))
    return h


@pytest.fixture()
def innovation_history_dir(tmp_path: Path) -> Path:
    """History directory with a NewProject innovation entry."""
    h = tmp_path / "history"
    h.mkdir()
    (h / "NewProject.json").write_text(json.dumps([
        {"date": "2026-07-08", "commits": 114, "mode": "INNOVATION",
         "improvements": [], "tests_passed": True, "email_status": "sent"},
    ]))
    return h


@pytest.fixture()
def single_row() -> pd.DataFrame:
    """Return a single-row DataFrame for property/amenity feature tests."""
    return pd.DataFrame({
        "bedrooms": [3],
        "bathrooms": [2.0],
        "year_built": [1995],
        "sqft": [1500],
        "price": [450000],
        "school_score": [7.0],
        "transit_score": [6.0],
        "walkability_score": [8.0],
        "crime_rate": [0.05],
        "renovation_year": [None],
    })
