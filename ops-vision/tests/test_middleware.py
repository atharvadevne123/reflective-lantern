"""Tests for correlation-ID, rate-limiting, and request-size middleware."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware import CorrelationIdMiddleware, RateLimitMiddleware, RequestSizeLimitMiddleware


def _build_app(middleware_cls, **kwargs) -> FastAPI:
    """Build a minimal app wrapped in the given middleware.

    Args:
        middleware_cls: Middleware class to install.
        **kwargs: Extra keyword arguments passed to the middleware.

    Returns:
        A FastAPI app exposing /ping and /health.
    """
    app = FastAPI()
    app.add_middleware(middleware_cls, **kwargs)

    @app.get("/ping")
    def ping():
        return {"pong": True}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


class TestCorrelationIdMiddleware:
    """Tests for CorrelationIdMiddleware."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Return a client for an app wrapped in CorrelationIdMiddleware."""
        return TestClient(_build_app(CorrelationIdMiddleware))

    def test_response_has_correlation_id_header(self, client):
        """Every response carries an X-Correlation-ID header."""
        resp = client.get("/ping")
        assert "X-Correlation-ID" in resp.headers

    def test_generated_id_is_non_empty(self, client):
        """A generated correlation ID is a non-empty string."""
        resp = client.get("/ping")
        assert len(resp.headers["X-Correlation-ID"]) > 0

    def test_incoming_correlation_id_is_preserved(self, client):
        """A caller-supplied correlation ID is echoed back unchanged."""
        supplied = "trace-abc-123"
        resp = client.get("/ping", headers={"X-Correlation-ID": supplied})
        assert resp.headers["X-Correlation-ID"] == supplied

    def test_generated_ids_are_unique_per_request(self, client):
        """Two requests without a supplied ID get distinct correlation IDs."""
        first = client.get("/ping").headers["X-Correlation-ID"]
        second = client.get("/ping").headers["X-Correlation-ID"]
        assert first != second


class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware."""

    def test_requests_under_limit_succeed(self):
        """Requests below the limit are all served normally."""
        client = TestClient(_build_app(RateLimitMiddleware, max_requests=5, window_seconds=60))
        for _ in range(5):
            assert client.get("/ping").status_code == 200

    def test_request_over_limit_returns_429(self):
        """The request past the limit is rejected with HTTP 429."""
        client = TestClient(_build_app(RateLimitMiddleware, max_requests=3, window_seconds=60))
        for _ in range(3):
            client.get("/ping")
        assert client.get("/ping").status_code == 429

    def test_429_includes_retry_after_header(self):
        """A throttled response tells the caller when to retry."""
        client = TestClient(_build_app(RateLimitMiddleware, max_requests=1, window_seconds=45))
        client.get("/ping")
        resp = client.get("/ping")
        assert resp.headers.get("Retry-After") == "45"

    def test_health_endpoint_is_exempt(self):
        """Health checks bypass the limiter so probes never throttle a service out."""
        client = TestClient(_build_app(RateLimitMiddleware, max_requests=2, window_seconds=60))
        for _ in range(10):
            assert client.get("/health").status_code == 200

    def test_limit_is_per_client_ip(self):
        """Exhausting one client's budget must not throttle a different IP."""
        client = TestClient(_build_app(RateLimitMiddleware, max_requests=2, window_seconds=60))
        for _ in range(2):
            client.get("/ping", headers={"X-Forwarded-For": "10.0.0.1"})

        assert client.get("/ping", headers={"X-Forwarded-For": "10.0.0.1"}).status_code == 429
        assert client.get("/ping", headers={"X-Forwarded-For": "10.0.0.2"}).status_code == 200

    def test_forwarded_for_takes_first_address(self):
        """X-Forwarded-For chains are keyed on the original client address."""
        client = TestClient(_build_app(RateLimitMiddleware, max_requests=1, window_seconds=60))
        client.get("/ping", headers={"X-Forwarded-For": "10.0.0.5, 172.16.0.1"})
        resp = client.get("/ping", headers={"X-Forwarded-For": "10.0.0.5, 192.168.1.1"})
        assert resp.status_code == 429

    def test_rate_limit_remaining_header_present(self):
        """Successful responses carry X-RateLimit-Remaining header."""
        client = TestClient(_build_app(RateLimitMiddleware, max_requests=5, window_seconds=60))
        resp = client.get("/ping")
        assert "X-RateLimit-Remaining" in resp.headers

    def test_rate_limit_remaining_decrements(self):
        """X-RateLimit-Remaining decreases with each request."""
        client = TestClient(_build_app(RateLimitMiddleware, max_requests=5, window_seconds=60))
        first = int(client.get("/ping").headers["X-RateLimit-Remaining"])
        second = int(client.get("/ping").headers["X-RateLimit-Remaining"])
        assert second < first

    def test_rate_limit_limit_header_matches_config(self):
        """X-RateLimit-Limit matches the configured max_requests."""
        client = TestClient(_build_app(RateLimitMiddleware, max_requests=7, window_seconds=60))
        resp = client.get("/ping")
        assert resp.headers["X-RateLimit-Limit"] == "7"

    def test_429_includes_rate_limit_remaining_zero(self):
        """A throttled 429 response sets X-RateLimit-Remaining to 0."""
        client = TestClient(_build_app(RateLimitMiddleware, max_requests=1, window_seconds=60))
        client.get("/ping")
        resp = client.get("/ping")
        assert resp.status_code == 429
        assert resp.headers.get("X-RateLimit-Remaining") == "0"


class TestRequestSizeLimitMiddleware:
    """Tests for RequestSizeLimitMiddleware."""

    def _build(self, max_bytes: int = 100) -> TestClient:
        app = FastAPI()
        app.add_middleware(RequestSizeLimitMiddleware, max_bytes=max_bytes)

        @app.post("/upload")
        def upload():
            return {"ok": True}

        return TestClient(app)

    def test_small_request_passes(self):
        """A request within the size limit is forwarded normally."""
        client = self._build(max_bytes=1024)
        resp = client.post("/upload", headers={"Content-Length": "512"})
        assert resp.status_code == 200

    def test_oversized_request_returns_413(self):
        """A request exceeding the limit is rejected with HTTP 413."""
        client = self._build(max_bytes=100)
        resp = client.post("/upload", headers={"Content-Length": "200"})
        assert resp.status_code == 413

    def test_413_response_has_detail_key(self):
        """The 413 response body contains a 'detail' key."""
        client = self._build(max_bytes=10)
        resp = client.post("/upload", headers={"Content-Length": "100"})
        assert "detail" in resp.json()

    def test_no_content_length_passes(self):
        """A request without Content-Length header is allowed through."""
        client = self._build(max_bytes=100)
        resp = client.post("/upload")
        assert resp.status_code == 200
