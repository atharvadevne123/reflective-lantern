"""Tests for correlation ID and rate limiting middleware."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


def test_health_returns_correlation_id(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "x-correlation-id" in resp.headers


def test_custom_correlation_id_echoed(client):
    custom_id = "test-corr-123"
    resp = client.get("/health", headers={"X-Correlation-ID": custom_id})
    assert resp.headers.get("x-correlation-id") == custom_id


def test_predict_returns_correlation_id(client, predict_payload):
    resp = client.post("/predict", json=predict_payload)
    assert "x-correlation-id" in resp.headers


@pytest.mark.parametrize("path", ["/health", "/version"])
def test_all_endpoints_return_correlation_id(client, path):
    resp = client.get(path)
    assert "x-correlation-id" in resp.headers
