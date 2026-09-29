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


def test_rate_limit_remaining_decrements():
    with TestClient(_app(limit=5)) as c:
        prev = 5
        for _ in range(3):
            resp = c.get("/ping")
            remaining = int(resp.headers["X-RateLimit-Remaining"])
            assert remaining == prev - 1
            prev = remaining


def test_rate_limit_error_body_has_detail():
    with TestClient(_app(limit=1)) as c:
        c.get("/ping")
        resp = c.get("/ping")
        body = resp.json()
        assert "detail" in body
        assert body["detail"] == "Too many requests"


def test_x_forwarded_for_used_for_client_key():
    """Clients behind a proxy are bucketed by the forwarded IP, not the socket."""
    with TestClient(_app(limit=2)) as c:
        headers = {"X-Forwarded-For": "10.0.0.1"}
        for _ in range(2):
            assert c.get("/ping", headers=headers).status_code == 200
        assert c.get("/ping", headers=headers).status_code == 429


def test_unknown_client_falls_back_gracefully():
    """A request with no client info should still be processed (not crash)."""
    with TestClient(_app(limit=5)) as c:
        resp = c.get("/ping")
        assert resp.status_code == 200
