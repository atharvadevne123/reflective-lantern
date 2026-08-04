"""Tests for custom ASGI middleware components."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_process_time_header_present(client: TestClient, sample_sensor_payload: dict) -> None:
    resp = client.post("/api/v1/predict", json=sample_sensor_payload)
    assert "x-process-time" in resp.headers


def test_process_time_header_on_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert "x-process-time" in resp.headers


def test_security_headers_present(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.headers.get("x-content-type-options") == "nosniff"
    assert resp.headers.get("x-frame-options") == "DENY"
    assert "x-xss-protection" in resp.headers


def test_security_headers_on_predict(client: TestClient, sample_sensor_payload: dict) -> None:
    resp = client.post("/api/v1/predict", json=sample_sensor_payload)
    assert resp.headers.get("x-content-type-options") == "nosniff"
    assert resp.headers.get("referrer-policy") == "no-referrer"


def test_correlation_id_auto_generated(client: TestClient, sample_sensor_payload: dict) -> None:
    resp = client.post("/api/v1/predict", json=sample_sensor_payload)
    assert "x-correlation-id" in resp.headers
    assert len(resp.headers["x-correlation-id"]) > 0


def test_process_time_is_numeric(client: TestClient) -> None:
    resp = client.get("/health")
    header_val = resp.headers.get("x-process-time", "")
    assert "ms" in header_val
    numeric_part = header_val.replace("ms", "")
    assert float(numeric_part) >= 0
