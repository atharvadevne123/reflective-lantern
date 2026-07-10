"""Shared pytest fixtures for the Reflective Lantern test suite."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Generator

import pytest

# Tests for the staged Cyber-Sentinel app (app/, pipelines/) need heavyweight
# ML dependencies that are not part of Reflective Lantern's own requirements.
# Skip collecting them when their primary dependency is absent so the core
# automation suite stays runnable everywhere.
_STAGED_TEST_DEPS: dict[str, str] = {
    "test_anomaly.py": "numpy",
    "test_api.py": "fastapi",
    "test_cache.py": "numpy",
    "test_database.py": "sqlalchemy",
    "test_faiss.py": "numpy",
    "test_features.py": "numpy",
    "test_investment.py": "numpy",
    "test_market_context.py": "numpy",
    "test_mlflow_stub.py": "numpy",
    "test_model.py": "numpy",
    "test_monitoring.py": "scipy",
    "test_time_series.py": "numpy",
}

collect_ignore: list[str] = [
    filename for filename, dep in _STAGED_TEST_DEPS.items() if importlib.util.find_spec(dep) is None
]


@pytest.fixture()
def history_dir(tmp_path: Path) -> Path:
    """Return a temporary history directory pre-populated with sample files."""
    h = tmp_path / "history"
    h.mkdir()
    sample: list[dict[str, Any]] = [
        {
            "date": "2026-06-01",
            "mode": "improvement",
            "commits": 60,
            "tests_passed": True,
            "improvements": ["added type annotations", "added docstrings"],
        },
        {
            "date": "2026-06-15",
            "mode": "improvement",
            "commits": 60,
            "tests_passed": True,
            "improvements": ["added pytest suite"],
        },
    ]
    (h / "SampleRepo.json").write_text(json.dumps(sample))
    (h / "AnotherRepo.json").write_text(json.dumps([sample[0]]))
    return h


@pytest.fixture()
def single_entry_history_dir(tmp_path: Path) -> Path:
    """Return a history directory with a single-dict-format file."""
    h = tmp_path / "history"
    h.mkdir()
    entry = {"last_run": "2026-06-20", "commits": 45, "mode": "improvement"}
    (h / "OldRepo.json").write_text(json.dumps(entry))
    return h


@pytest.fixture()
def invalid_history_dir(tmp_path: Path) -> Path:
    """Return a history directory with an invalid JSON file."""
    h = tmp_path / "history"
    h.mkdir()
    (h / "Broken.json").write_text("{not valid json")
    return h


@pytest.fixture()
def env_with_pat(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Monkeypatch GH_PAT so scripts don't require a real token."""
    monkeypatch.setenv("GH_PAT", "test-token-123")
    monkeypatch.setenv("GITHUB_USERNAME", "test-owner")
    yield
    # teardown handled by monkeypatch


@pytest.fixture()
def settings_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Set all required env vars for Settings."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("GH_PAT", "ghp_test")
    monkeypatch.setenv("NOTION_API_KEY", "secret_test")
    monkeypatch.setenv("GMAIL_USER", "test@gmail.com")
    monkeypatch.setenv("GMAIL_APP_PASS", "test-pass")
    yield


@pytest.fixture()
def multi_repo_history_dir(tmp_path: Path) -> Path:
    """Return a history directory with multiple repos across different modes."""
    h = tmp_path / "history"
    h.mkdir()
    (h / "Alpha.json").write_text(
        json.dumps(
            [
                {"date": "2026-07-01", "mode": "improvement", "commits": 60, "tests_passed": True},
                {"date": "2026-07-03", "mode": "improvement", "commits": 60, "tests_passed": True},
            ]
        )
    )
    (h / "Beta.json").write_text(
        json.dumps(
            [
                {"date": "2026-07-08", "mode": "innovation", "commits": 120, "tests_passed": True},
            ]
        )
    )
    (h / "Gamma.json").write_text(
        json.dumps(
            [
                {"date": "2026-07-02", "mode": "improvement", "commits": 60, "tests_passed": False},
            ]
        )
    )
    return h


@pytest.fixture()
def db_session():  # type: ignore[return]
    """Yield an in-memory SQLite session and roll it back after each test."""
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from app.database import Base

        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = Session()
        try:
            yield session
        finally:
            session.close()
            Base.metadata.drop_all(bind=engine)
    except ImportError:
        pytest.skip("sqlalchemy not installed")


@pytest.fixture()
def sample_df():  # type: ignore[return]
    """Return a small synthetic property DataFrame for ML tests."""
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        pytest.skip("numpy/pandas not installed")
        return
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame(
        {
            "sqft": rng.uniform(800, 4000, n),
            "bedrooms": rng.integers(1, 6, n),
            "bathrooms": rng.uniform(1, 4, n),
            "lot_size": rng.uniform(2000, 15000, n),
            "year_built": rng.integers(1960, 2020, n),
            "condition_score": rng.uniform(3, 9, n),
            "school_score": rng.uniform(4, 9, n),
            "transit_score": rng.uniform(3, 9, n),
            "walkability_score": rng.uniform(3, 9, n),
            "crime_rate": rng.uniform(0.05, 0.6, n),
            "median_neighborhood_price": rng.uniform(200_000, 800_000, n),
            "median_price_per_sqft": rng.uniform(100, 400, n),
            "avg_rental_yield": rng.uniform(0.03, 0.12, n),
            "listing_days": rng.integers(1, 180, n),
            "list_price": rng.uniform(200_000, 900_000, n),
        }
    )


@pytest.fixture()
def sample_target(sample_df):  # type: ignore[return]
    """Return a synthetic target array matching sample_df."""
    try:
        import numpy as np
    except ImportError:
        pytest.skip("numpy not installed")
        return
    rng = np.random.default_rng(42)
    return rng.uniform(150_000, 1_200_000, len(sample_df))


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):  # type: ignore[return]
    """Return a FastAPI TestClient with an in-memory DB and synthetic model."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")
        return
    os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test")
    from app.database import Base, engine
    from app.main import app

    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def sample_property() -> dict[str, Any]:
    """Return a valid sample property dict for API request payloads."""
    return {
        "sqft": 1800.0,
        "bedrooms": 3,
        "bathrooms": 2.0,
        "lot_size": 6000.0,
        "year_built": 1995,
        "condition_score": 7.0,
        "zipcode": "94102",
        "city": "San Francisco",
        "state": "CA",
        "school_score": 7.0,
        "transit_score": 8.0,
        "walkability_score": 8.0,
        "crime_rate": 0.25,
        "median_neighborhood_price": 900_000.0,
        "median_price_per_sqft": 750.0,
        "avg_rental_yield": 0.04,
        "listing_days": 14,
        "list_price": 950_000.0,
    }


@pytest.fixture()
def innovation_history_dir(tmp_path: Path) -> Path:
    """Return a history directory with an INNOVATION mode entry."""
    h = tmp_path / "history"
    h.mkdir()
    (h / "NewProject.json").write_text(
        json.dumps(
            [
                {
                    "date": "2026-07-08",
                    "mode": "innovation",
                    "commits": 114,
                    "tests_passed": True,
                    "improvements": ["built new ML pipeline from scratch"],
                }
            ]
        )
    )
    return h
