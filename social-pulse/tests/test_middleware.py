"""Tests for application middleware."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware import ContentSecurityPolicyMiddleware, CorrelationIdMiddleware, SecurityHeadersMiddleware


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


def _make_csp_app(policy: str | None = None) -> FastAPI:
    app = FastAPI()
    app.add_middleware(ContentSecurityPolicyMiddleware, policy=policy)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    return app


class TestContentSecurityPolicyMiddleware:
    def test_default_policy_header_present(self) -> None:
        client = TestClient(_make_csp_app())
        resp = client.get("/ping")
        assert "Content-Security-Policy" in resp.headers

    def test_default_policy_value(self) -> None:
        client = TestClient(_make_csp_app())
        resp = client.get("/ping")
        assert "default-src 'self'" in resp.headers["Content-Security-Policy"]

    def test_custom_policy_applied(self) -> None:
        custom = "default-src 'none'; img-src 'self'"
        client = TestClient(_make_csp_app(policy=custom))
        resp = client.get("/ping")
        assert resp.headers["Content-Security-Policy"] == custom

    def test_script_src_in_default(self) -> None:
        client = TestClient(_make_csp_app())
        resp = client.get("/ping")
        assert "script-src" in resp.headers["Content-Security-Policy"]

    def test_object_src_none_in_default(self) -> None:
        client = TestClient(_make_csp_app())
        resp = client.get("/ping")
        assert "object-src 'none'" in resp.headers["Content-Security-Policy"]
