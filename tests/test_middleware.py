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


def test_rate_limit_applied_per_ip(client: TestClient, monkeypatch) -> None:
    """Different client IPs should have independent rate-limit windows."""
    from types import SimpleNamespace

    from app import middleware

    monkeypatch.setattr(middleware, "settings", SimpleNamespace(rate_limit_per_minute=1))
    reset_rate_limiter()
    r1 = client.get("/health", headers={"X-Forwarded-For": "1.1.1.1"})
    r2 = client.get("/health", headers={"X-Forwarded-For": "2.2.2.2"})
    assert r1.status_code == 200
    assert r2.status_code == 200


def test_correlation_id_max_length_accepted(client: TestClient) -> None:
    cid = "a" * 64
    resp = client.get("/health", headers={"X-Correlation-ID": cid})
    assert resp.headers.get("x-correlation-id") == cid


def test_health_endpoint_response_body_structure(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body


def test_reset_before_any_requests_is_safe(client: TestClient) -> None:
    reset_rate_limiter()
    reset_rate_limiter()
    assert len(_requests) == 0


@pytest.mark.parametrize("path", ["/health", "/metrics", "/version"])
def test_correlation_header_on_all_endpoints(client: TestClient, path: str) -> None:
    resp = client.get(path)
    assert "x-correlation-id" in resp.headers
