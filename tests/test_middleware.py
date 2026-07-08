"""Tests for the rate limiting middleware."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.middleware import _requests, reset_rate_limiter


@pytest.fixture(autouse=True)
def clean_rate_limiter() -> None:
    reset_rate_limiter()


def test_requests_under_limit_pass(client: TestClient) -> None:
    for _ in range(5):
        assert client.get("/health").status_code == 200


def test_rate_limit_returns_429_when_exceeded(client: TestClient, monkeypatch) -> None:
    from types import SimpleNamespace

    from app import middleware

    monkeypatch.setattr(middleware, "settings", SimpleNamespace(rate_limit_per_minute=3))
    codes = [client.get("/health").status_code for _ in range(5)]
    assert 429 in codes


def test_rate_limit_response_has_retry_after(client: TestClient, monkeypatch) -> None:
    from types import SimpleNamespace

    from app import middleware

    monkeypatch.setattr(middleware, "settings", SimpleNamespace(rate_limit_per_minute=1))
    client.get("/health")
    resp = client.get("/health")
    assert resp.status_code == 429
    assert resp.headers.get("retry-after") == "60"


def test_reset_clears_windows(client: TestClient) -> None:
    client.get("/health")
    assert len(_requests) >= 0
    reset_rate_limiter()
    assert len(_requests) == 0
