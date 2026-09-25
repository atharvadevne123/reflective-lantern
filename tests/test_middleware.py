"""Tests for rate limiting and correlation-ID middleware."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware import RateLimitMiddleware


def _app(limit: int) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, limit=limit)

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    return app


def test_requests_under_limit_pass():
    with TestClient(_app(limit=5)) as c:
        for _ in range(5):
            assert c.get("/ping").status_code == 200


def test_request_over_limit_rejected():
    with TestClient(_app(limit=3)) as c:
        for _ in range(3):
            assert c.get("/ping").status_code == 200
        resp = c.get("/ping")
        assert resp.status_code == 429
        assert resp.json()["error"] == "RateLimitExceeded"


def test_rate_limit_headers_present():
    with TestClient(_app(limit=10)) as c:
        resp = c.get("/ping")
        assert resp.headers["X-RateLimit-Limit"] == "10"
        assert int(resp.headers["X-RateLimit-Remaining"]) == 9


def test_retry_after_header_on_429():
    with TestClient(_app(limit=1)) as c:
        c.get("/ping")
        resp = c.get("/ping")
        assert resp.status_code == 429
        assert int(resp.headers["Retry-After"]) >= 1


def test_correlation_id_header_returned(client):
    resp = client.get("/api/v1/health")
    assert "X-Request-ID" in resp.headers
    assert "X-Response-Time-Ms" in resp.headers


def test_correlation_id_is_echoed(client):
    resp = client.get("/api/v1/health", headers={"X-Request-ID": "trace-123"})
    assert resp.headers["X-Request-ID"] == "trace-123"


import pytest  # noqa: E402


@pytest.mark.parametrize("limit", [1, 5, 10])
def test_exactly_limit_requests_allowed(limit: int) -> None:
    """Exactly `limit` requests succeed; the (limit+1)-th is rejected with 429."""
    with TestClient(_app(limit=limit)) as c:
        for _ in range(limit):
            assert c.get("/ping").status_code == 200
        assert c.get("/ping").status_code == 429


@pytest.mark.parametrize("limit,n_ok", [(3, 2), (5, 4), (10, 9)])
def test_remaining_header_decrements(limit: int, n_ok: int) -> None:
    """X-RateLimit-Remaining decrements correctly after successive requests."""
    with TestClient(_app(limit=limit)) as c:
        for i in range(n_ok):
            resp = c.get("/ping")
            assert int(resp.headers["X-RateLimit-Remaining"]) == limit - i - 1


@pytest.mark.parametrize("limit", [2, 4, 8])
def test_limit_header_is_constant(limit: int) -> None:
    """X-RateLimit-Limit always reflects the configured limit."""
    with TestClient(_app(limit=limit)) as c:
        for _ in range(min(limit, 3)):
            resp = c.get("/ping")
            assert resp.headers["X-RateLimit-Limit"] == str(limit)
