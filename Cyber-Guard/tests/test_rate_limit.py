"""Rate limiting middleware tests for Cyber-Guard."""

from __future__ import annotations

import time
from collections import deque

import pytest

from app.rate_limit import WINDOW_SECONDS, _prune, reset_rate_limit_state


@pytest.fixture(autouse=True)
def clear_state():
    reset_rate_limit_state()
    yield
    reset_rate_limit_state()


def test_prune_drops_expired_timestamps():
    now = time.time()
    bucket = deque([now - WINDOW_SECONDS - 10, now - WINDOW_SECONDS - 1, now - 5])
    _prune(bucket, now)
    assert len(bucket) == 1


def test_prune_keeps_all_recent():
    now = time.time()
    bucket = deque([now - 5, now - 3, now - 1])
    _prune(bucket, now)
    assert len(bucket) == 3


def test_prune_on_empty_bucket():
    bucket: deque[float] = deque()
    _prune(bucket, time.time())
    assert len(bucket) == 0


def test_rate_limit_headers_present(client):
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    assert "X-RateLimit-Limit" in resp.headers
    assert "X-RateLimit-Remaining" in resp.headers


def test_health_endpoint_is_exempt(client):
    """Liveness probes must not be able to rate-limit the service out."""
    for _ in range(50):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
    assert "X-RateLimit-Limit" not in resp.headers


def test_limit_enforced_returns_429(client, monkeypatch):
    """Once the window fills, further calls get 429 with a Retry-After."""
    from app import rate_limit

    # Shrink the limit so the test does not need 120 real requests.
    for mw in client.app.user_middleware:
        if mw.cls is rate_limit.RateLimitMiddleware:
            mw.kwargs["limit_per_minute"] = 3
    client.app.middleware_stack = client.app.build_middleware_stack()
    reset_rate_limit_state()

    codes = [client.get("/api/v1/metrics").status_code for _ in range(5)]
    assert 429 in codes, f"expected a 429 once over limit, got {codes}"

    limited = client.get("/api/v1/metrics")
    assert limited.status_code == 429
    assert "Retry-After" in limited.headers
    body = limited.json()
    assert body["detail"] == "rate limit exceeded"
    assert body["retry_after_seconds"] >= 1
