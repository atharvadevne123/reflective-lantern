"""Shared pytest fixtures for Cyber-Sentinel test suite."""

from __future__ import annotations

import os

# Use an in-file SQLite database for tests; must be set before app.database
# builds its engine at import time.
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/cyber_sentinel_test.db")

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Provide a FastAPI TestClient for end-to-end API tests."""
    from app.main import app

    return TestClient(app)


@pytest.fixture(scope="session")
def trained_client(client: TestClient) -> TestClient:
    """Train the model and return a ready client."""
    resp = client.post("/train")
    assert resp.status_code == 200
    return client


@pytest.fixture
def sample_event() -> dict[str, object]:
    """Return a minimal valid network event payload."""
    return {
        "protocol": "tcp",
        "payload_bytes": 512,
        "duration_ms": 120.0,
        "src_bytes": 1024,
        "dst_bytes": 2048,
        "flags": "SA",
        "count": 10,
        "srv_count": 8,
        "serror_rate": 0.1,
        "same_srv_rate": 0.9,
    }


@pytest.fixture
def dos_event() -> dict[str, object]:
    """Return a network event with DoS-like characteristics."""
    return {
        "protocol": "tcp",
        "payload_bytes": 65535,
        "duration_ms": 0.1,
        "src_bytes": 500000,
        "dst_bytes": 0,
        "flags": "S",
        "count": 511,
        "srv_count": 511,
        "serror_rate": 0.99,
        "same_srv_rate": 0.99,
        "diff_srv_rate": 0.01,
    }


@pytest.fixture(autouse=True)
def reset_model_cache() -> None:
    """Clear the in-memory model cache before each test."""
    from app.model import clear_model_cache
    from app.monitoring import reset_monitoring

    clear_model_cache()
    reset_monitoring()
