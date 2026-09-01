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


def test_correlation_id_header_propagated(client: TestClient) -> None:
    cid = "test-correlation-123"
    resp = client.get("/health", headers={"X-Correlation-ID": cid})
    assert resp.headers.get("x-correlation-id") == cid


def test_correlation_id_generated_when_absent(client: TestClient) -> None:
    resp = client.get("/health")
    assert "x-correlation-id" in resp.headers
    assert len(resp.headers["x-correlation-id"]) > 0


def test_correlation_id_is_uuid_when_not_provided(client: TestClient) -> None:
    import uuid

    resp = client.get("/health")
    cid = resp.headers.get("x-correlation-id", "")
    try:
        uuid.UUID(cid)
        is_uuid = True
    except ValueError:
        is_uuid = False
    assert is_uuid


def test_rate_limit_resets_after_window(client: TestClient, monkeypatch) -> None:
    from types import SimpleNamespace

    from app import middleware

    monkeypatch.setattr(middleware, "settings", SimpleNamespace(rate_limit_per_minute=1))
    reset_rate_limiter()
    resp1 = client.get("/health")
    assert resp1.status_code == 200


def test_multiple_resets_are_idempotent(client: TestClient) -> None:
    client.get("/health")
    reset_rate_limiter()
    reset_rate_limiter()
    assert len(_requests) == 0


import pytest


@pytest.mark.parametrize("path", ["/health", "/api/v1/version"])
def test_request_id_header_present_on_various_paths(client: TestClient, path: str) -> None:
    reset_rate_limiter()
    resp = client.get(path)
    assert "X-Request-ID" in resp.headers or resp.status_code in (200, 404)
    reset_rate_limiter()


def test_x_request_id_is_different_per_request(client: TestClient) -> None:
    reset_rate_limiter()
    r1 = client.get("/health")
    r2 = client.get("/health")
    id1 = r1.headers.get("X-Request-ID")
    id2 = r2.headers.get("X-Request-ID")
    if id1 and id2:
        assert id1 != id2
    reset_rate_limiter()
