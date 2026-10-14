"""Tests for rate limiting and correlation ID middleware."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_correlation_id_generated_when_absent(client):
    resp = client.get("/api/v1/health")
    assert "x-correlation-id" in resp.headers
    assert len(resp.headers["x-correlation-id"]) > 8


def test_correlation_id_echoed_when_provided(client):
    cid = "test-correlation-abc123"
    resp = client.get("/api/v1/health", headers={"X-Correlation-ID": cid})
    assert resp.headers.get("x-correlation-id") == cid


def test_health_endpoint_returns_200(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200


def test_health_response_has_status_key(client):
    resp = client.get("/api/v1/health")
    body = resp.json()
    assert "status" in body


def test_metrics_endpoint_returns_200(client):
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200


def test_metrics_response_has_expected_keys(client):
    resp = client.get("/api/v1/metrics")
    body = resp.json()
    assert "total_predictions" in body
    assert "uptime_seconds" in body


def test_invalid_route_returns_404(client):
    resp = client.get("/api/v1/nonexistent")
    assert resp.status_code == 404


def test_drift_endpoint_requires_post(client):
    resp = client.get("/api/v1/drift")
    assert resp.status_code == 405


def test_drift_post_with_missing_body_returns_422(client):
    resp = client.post("/api/v1/drift", json={})
    assert resp.status_code == 422
