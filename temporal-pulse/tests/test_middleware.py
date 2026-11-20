"""Tests for rate limiting middleware."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from fastapi.testclient import TestClient


def make_app(limit: int = 3, window: int = 60) -> TestClient:
    """Build a minimal app with a tight rate limit for testing."""
    from app.middleware import RateLimitMiddleware

    test_app = FastAPI()
    test_app.add_middleware(RateLimitMiddleware, requests_per_window=limit, window_seconds=window)

    @test_app.get("/ping")
    async def ping():
        return {"pong": True}

    return TestClient(test_app)


class TestRateLimit:
    def test_requests_under_limit_pass(self):
        client = make_app(limit=5)
        for _ in range(5):
            assert client.get("/ping").status_code == 200

    def test_request_over_limit_rejected(self):
        client = make_app(limit=3)
        for _ in range(3):
            client.get("/ping")
        resp = client.get("/ping")
        assert resp.status_code == 429

    def test_429_has_retry_after_header(self):
        client = make_app(limit=1)
        client.get("/ping")
        resp = client.get("/ping")
        assert "Retry-After" in resp.headers

    def test_forwarded_for_header_used(self):
        client = make_app(limit=1)
        client.get("/ping", headers={"X-Forwarded-For": "1.2.3.4"})
        # Different client IP gets its own bucket
        resp = client.get("/ping", headers={"X-Forwarded-For": "5.6.7.8"})
        assert resp.status_code == 200

    def test_same_forwarded_ip_shares_bucket(self):
        client = make_app(limit=1)
        client.get("/ping", headers={"X-Forwarded-For": "9.9.9.9"})
        resp = client.get("/ping", headers={"X-Forwarded-For": "9.9.9.9"})
        assert resp.status_code == 429
