"""Rate limiting middleware tests."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.rate_limit import RateLimitMiddleware


def _make_app(limit: int) -> TestClient:
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, requests_per_minute=limit)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    return TestClient(app)


def test_rate_limit_allows_requests_within_limit():
    client = _make_app(limit=10)
    for _ in range(5):
        resp = client.get("/ping")
        assert resp.status_code == 200


def test_rate_limit_blocks_after_limit():
    client = _make_app(limit=3)
    for _ in range(3):
        client.get("/ping")
    # 4th request should be blocked
    resp = client.get("/ping")
    assert resp.status_code == 429


def test_rate_limit_response_body():
    client = _make_app(limit=1)
    client.get("/ping")
    resp = client.get("/ping")
    assert resp.status_code == 429
    assert "Rate limit" in resp.json()["detail"]
