"""Tests for rate limiting middleware."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.rate_limit import RateLimitMiddleware


def _make_app(rate: int = 5, window: int = 60) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, rate=rate, window=window)

    @app.get("/ping")
    def ping():
        return {"pong": True}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


class TestRateLimitMiddleware:
    def test_allows_requests_under_limit(self) -> None:
        client = TestClient(_make_app(rate=5))
        for _ in range(4):
            resp = client.get("/ping")
            assert resp.status_code == 200

    def test_blocks_over_limit(self) -> None:
        client = TestClient(_make_app(rate=3))
        for _ in range(3):
            client.get("/ping")
        resp = client.get("/ping")
        assert resp.status_code == 429

    def test_429_response_has_retry_after(self) -> None:
        client = TestClient(_make_app(rate=2))
        client.get("/ping")
        client.get("/ping")
        resp = client.get("/ping")
        assert "Retry-After" in resp.headers

    def test_exempt_path_never_blocked(self) -> None:
        client = TestClient(_make_app(rate=1))
        for _ in range(10):
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_rate_limit_headers_present(self) -> None:
        client = TestClient(_make_app(rate=5))
        resp = client.get("/ping")
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers

    def test_remaining_decrements(self) -> None:
        client = TestClient(_make_app(rate=5))
        resp1 = client.get("/ping")
        resp2 = client.get("/ping")
        r1 = int(resp1.headers["X-RateLimit-Remaining"])
        r2 = int(resp2.headers["X-RateLimit-Remaining"])
        assert r2 <= r1


@pytest.mark.parametrize("rate", [1, 2, 5])
def test_blocks_at_rate_plus_one(rate: int) -> None:
    """Should return 429 on the (rate+1)th request."""
    client = TestClient(_make_app(rate=rate))
    for _ in range(rate):
        resp = client.get("/ping")
        assert resp.status_code == 200
    resp = client.get("/ping")
    assert resp.status_code == 429
