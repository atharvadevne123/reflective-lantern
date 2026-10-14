"""Tests for application middleware."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware import CorrelationIdMiddleware, SecurityHeadersMiddleware


def _make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/echo")
    def echo():
        return {"ok": True}

    return app


@pytest.fixture
def middleware_client() -> TestClient:
    return TestClient(_make_app())


class TestCorrelationIdMiddleware:
    def test_response_has_request_id(self, middleware_client: TestClient) -> None:
        resp = middleware_client.get("/echo")
        assert "X-Request-ID" in resp.headers

    def test_response_has_latency_header(self, middleware_client: TestClient) -> None:
        resp = middleware_client.get("/echo")
        assert "X-Response-Time-Ms" in resp.headers

    def test_latency_is_positive_float(self, middleware_client: TestClient) -> None:
        resp = middleware_client.get("/echo")
        latency = float(resp.headers["X-Response-Time-Ms"])
        assert latency >= 0.0

    def test_custom_request_id_echoed(self, middleware_client: TestClient) -> None:
        resp = middleware_client.get("/echo", headers={"X-Request-ID": "my-id-42"})
        assert resp.headers["X-Request-ID"] == "my-id-42"

    def test_auto_generated_id_is_uuid(self, middleware_client: TestClient) -> None:
        import uuid
        resp = middleware_client.get("/echo")
        rid = resp.headers["X-Request-ID"]
        # Should parse as valid UUID
        uuid.UUID(rid)


class TestSecurityHeadersMiddleware:
    def test_x_content_type_options(self, middleware_client: TestClient) -> None:
        resp = middleware_client.get("/echo")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"

    def test_x_frame_options(self, middleware_client: TestClient) -> None:
        resp = middleware_client.get("/echo")
        assert resp.headers.get("X-Frame-Options") == "DENY"

    def test_referrer_policy(self, middleware_client: TestClient) -> None:
        resp = middleware_client.get("/echo")
        assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"


@pytest.mark.parametrize("path", ["/echo", "/echo"])
def test_middleware_applied_to_all_paths(middleware_client: TestClient, path: str) -> None:
    resp = middleware_client.get(path)
    assert "X-Request-ID" in resp.headers
    assert "X-Content-Type-Options" in resp.headers
