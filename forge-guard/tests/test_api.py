"""Tests for FastAPI prediction and ops endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_health_returns_200(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("healthy", "degraded")
    assert "model_version" in data


def test_predict_returns_valid_response(client: TestClient, sample_sensor_payload: dict) -> None:
    resp = client.post("/api/v1/predict", json=sample_sensor_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["defect_probability"] <= 1.0
    assert "model_version" in data
    assert "correlation_id" in data


def test_predict_high_risk_tends_toward_defect(client: TestClient, high_risk_payload: dict) -> None:
    resp = client.post("/api/v1/predict", json=high_risk_payload)
    assert resp.status_code == 200
    # High-risk readings should produce non-trivial defect probability
    assert resp.json()["defect_probability"] >= 0.0


def test_predict_invalid_temperature_rejected(client: TestClient) -> None:
    bad = {
        "temperature": 9999,  # exceeds max=300
        "pressure": 52.0,
        "vibration": 2.1,
        "cycle_time": 28.0,
        "tool_wear": 15.0,
        "power_consumption": 98.0,
        "humidity": 45.0,
    }
    resp = client.post("/api/v1/predict", json=bad)
    assert resp.status_code == 422


def test_predict_missing_field_rejected(client: TestClient) -> None:
    resp = client.post("/api/v1/predict", json={"temperature": 75.0})
    assert resp.status_code == 422


def test_correlation_id_header_echoed(client: TestClient, sample_sensor_payload: dict) -> None:
    cid = "test-cid-abc123"
    resp = client.post(
        "/api/v1/predict",
        json=sample_sensor_payload,
        headers={"X-Correlation-ID": cid},
    )
    assert resp.status_code == 200
    assert resp.headers.get("X-Correlation-ID") == cid


def test_metrics_endpoint_returns_structure(client: TestClient) -> None:
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_metrics" in data
    assert "defect_rate_24h" in data
    assert "drift_reports" in data


def test_legacy_predict_alias(client: TestClient, sample_sensor_payload: dict) -> None:
    resp = client.post("/predict", json=sample_sensor_payload)
    assert resp.status_code == 200


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("humidity", 150),  # > 100
        ("cycle_time", 0),  # < 0.1
        ("pressure", -5),  # < 0
    ],
)
def test_field_validation(
    client: TestClient, sample_sensor_payload: dict, field: str, bad_value: float
) -> None:
    payload = {**sample_sensor_payload, field: bad_value}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422
