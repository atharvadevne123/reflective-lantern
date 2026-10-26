"""Middleware tests for Ticket-Oracle."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware import CORRELATION_ID_HEADER, CorrelationIDMiddleware, RateLimitHeaderMiddleware


@pytest.fixture()
def app_with_middleware():
    app = FastAPI()
    app.add_middleware(CorrelationIDMiddleware)
    app.add_middleware(RateLimitHeaderMiddleware)

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    @app.get("/rate-limited")
    async def rate_limited():
        from starlette.responses import Response
        return Response(status_code=429, content="Too Many Requests")

    return app


@pytest.fixture()
def client(app_with_middleware):
    return TestClient(app_with_middleware, raise_server_exceptions=False)


def test_correlation_id_generated_if_absent(client):
    resp = client.get("/ping")
    assert CORRELATION_ID_HEADER in resp.headers
    assert len(resp.headers[CORRELATION_ID_HEADER]) == 36  # UUID4 format


def test_correlation_id_propagated(client):
    cid = "test-correlation-id-123"
    resp = client.get("/ping", headers={CORRELATION_ID_HEADER: cid})
    assert resp.headers[CORRELATION_ID_HEADER] == cid


def test_each_request_unique_correlation_id(client):
    r1 = client.get("/ping")
    r2 = client.get("/ping")
    assert r1.headers[CORRELATION_ID_HEADER] != r2.headers[CORRELATION_ID_HEADER]


def test_rate_limit_adds_retry_after(client):
    resp = client.get("/rate-limited")
    assert resp.status_code == 429
    assert resp.headers.get("Retry-After") == "60"


def test_normal_response_no_retry_after(client):
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert "Retry-After" not in resp.headers
