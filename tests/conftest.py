"""Pytest fixtures and test database setup."""

from __future__ import annotations

import json as _json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app

TEST_DB_URL = "sqlite:///./test_watt_guard.db"

test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def create_test_db() -> None:
    """Create test DB tables once per session and drop them after."""
    Base.metadata.create_all(bind=test_engine)
    yield  # type: ignore[misc]
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session():
    """Yield a fresh SQLAlchemy session for each test."""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session):
    """FastAPI TestClient wired to the in-memory test database."""
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
def sample_df() -> pd.DataFrame:
    """300-row synthetic energy DataFrame for model tests."""
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame(
        {
            "hour": rng.integers(0, 24, n),
            "day_of_week": rng.integers(0, 7, n),
            "month": rng.integers(1, 13, n),
            "temperature_c": rng.uniform(5, 35, n),
            "humidity_pct": rng.uniform(30, 80, n),
            "occupancy": rng.integers(0, 150, n),
            "hvac_state": rng.integers(0, 2, n),
            "consumption_kwh": 8 + rng.normal(0, 3, n).clip(-7),
        }
    )


@pytest.fixture
def sample_y(sample_df: pd.DataFrame) -> pd.Series:
    return sample_df["consumption_kwh"].copy()


@pytest.fixture
def trained_model(sample_df, sample_y):
    from app.model import train_model

    bundle, metrics = train_model(sample_df, sample_y)
    return bundle, metrics


@pytest.fixture
def trained_anomaly_model(sample_df):
    from app.model import train_anomaly_model

    return train_anomaly_model(sample_df)


@pytest.fixture
def energy_payload():
    return {
        "building_id": "bldg-001",
        "timestamp": "2025-06-01T14:00:00",
        "hour": 14,
        "day_of_week": 1,
        "month": 6,
        "temperature_c": 28.5,
        "humidity_pct": 60.0,
        "occupancy": 50,
        "hvac_state": 1,
        "consumption_kwh": 12.0,
    }


@pytest.fixture
def sample_target(sample_df: pd.DataFrame) -> pd.Series:
    """Target series for model training tests (alias for sample_y)."""
    return sample_df["consumption_kwh"].copy()


@pytest.fixture
def settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set all required environment variables for Settings tests."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("GH_PAT", "ghp_test")
    monkeypatch.setenv("NOTION_API_KEY", "secret_test")
    monkeypatch.setenv("GMAIL_USER", "test@gmail.com")
    monkeypatch.setenv("GMAIL_APP_PASS", "test-app-pass")
    monkeypatch.setenv("REPORT_RECIPIENT", "test@gmail.com")
    import config.settings as cs
    cs._settings = None


@pytest.fixture
def history_dir(tmp_path: Path) -> Path:
    """A temp history dir with SampleRepo (2 entries) and AnotherRepo (1 entry)."""
    h = tmp_path / "history"
    h.mkdir()
    (h / "SampleRepo.json").write_text(
        _json.dumps([
            {
                "date": "2026-01-01",
                "commits": 10,
                "mode": "improvement",
                "improvements": [{"id": 1}, {"id": 2}],
                "tests_passed": True,
                "email_status": "sent",
            },
            {
                "date": "2026-02-01",
                "commits": 5,
                "mode": "improvement",
                "improvements": [],
                "tests_passed": False,
                "email_status": "",
            },
        ])
    )
    (h / "AnotherRepo.json").write_text(
        _json.dumps([
            {
                "date": "2026-01-15",
                "commits": 3,
                "mode": "improvement",
                "improvements": [{"id": 1}],
                "tests_passed": True,
                "email_status": "",
            }
        ])
    )
    return h


@pytest.fixture
def single_entry_history_dir(tmp_path: Path) -> Path:
    """A temp history dir with one repo having a last_run fallback date."""
    h = tmp_path / "history_single"
    h.mkdir()
    (h / "OneRepo.json").write_text(
        _json.dumps([{"last_run": "2026-06-20", "commits": 5, "mode": "improvement"}])
    )
    return h


@pytest.fixture
def invalid_history_dir(tmp_path: Path) -> Path:
    """A temp history dir containing only malformed JSON."""
    h = tmp_path / "history_invalid"
    h.mkdir()
    (h / "Bad.json").write_text("not valid json {{{")
    return h


@pytest.fixture
def multi_repo_history_dir(tmp_path: Path) -> Path:
    """A temp history dir with Alpha (2 runs, 120 commits), Beta (120), Gamma (60) repos."""
    h = tmp_path / "history_multi"
    h.mkdir()
    (h / "Alpha.json").write_text(
        _json.dumps([
            {"date": "2026-07-01", "commits": 60, "mode": "IMPROVEMENT", "tests_passed": True},
            {"date": "2026-07-08", "commits": 60, "mode": "Innovation", "tests_passed": True},
        ])
    )
    (h / "Beta.json").write_text(
        _json.dumps([
            {"date": "2026-07-05", "commits": 120, "mode": "improvement", "tests_passed": True}
        ])
    )
    (h / "Gamma.json").write_text(
        _json.dumps([
            {"date": "2026-07-08", "commits": 60, "mode": "innovation", "tests_passed": False}
        ])
    )
    return h


@pytest.fixture
def innovation_history_dir(tmp_path: Path) -> Path:
    """A temp history dir with NewProject in innovation mode."""
    h = tmp_path / "history_innovation"
    h.mkdir()
    (h / "NewProject.json").write_text(
        _json.dumps([
            {"date": "2026-07-08", "commits": 114, "mode": "innovation", "tests_passed": True},
        ])
    )
    return h


@pytest.fixture
def single_row() -> pd.DataFrame:
    """A single-row DataFrame with property and amenity columns for feature tests."""
    return pd.DataFrame(
        {
            "bedrooms": [3],
            "bathrooms": [2.0],
            "year_built": [1990],
            "school_score": [7.0],
            "transit_score": [6.0],
            "walkability_score": [8.0],
            "crime_rate": [0.2],
            "hour": [12],
            "day_of_week": [1],
            "month": [6],
            "temperature_c": [22.0],
            "humidity_pct": [55.0],
            "occupancy": [50],
            "hvac_state": [1],
        }
    )
