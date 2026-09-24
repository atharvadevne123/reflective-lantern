"""Tests for FastAPI endpoints."""
from __future__ import annotations

import pytest


def test_health_returns_200(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("healthy", "degraded")
    assert "model_version" in data


def test_metrics_returns_200(client):
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    assert "model_version" in resp.json()


def test_predict_valid_payload(client, predict_payload):
    resp = client.post("/api/v1/predict", json=predict_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_minutes" in data
    assert data["predicted_minutes"] > 0
    assert "predicted_hours" in data
    assert data["model_version"] is not None


def test_predict_returns_hours_consistent(client, predict_payload):
    resp = client.post("/api/v1/predict", json=predict_payload)
    data = resp.json()
    assert abs(data["predicted_hours"] - data["predicted_minutes"] / 60) < 0.01


def test_predict_invalid_carrier(client, predict_payload):
    payload = {**predict_payload, "carrier": "UNKNOWN_CARRIER"}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_predict_invalid_route_type(client, predict_payload):
    payload = {**predict_payload, "route_type": "space"}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_predict_negative_distance(client, predict_payload):
    payload = {**predict_payload, "distance_km": -10}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_predict_zero_weight(client, predict_payload):
    payload = {**predict_payload, "weight_kg": 0}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


@pytest.mark.parametrize("carrier", ["DHL", "FedEx", "UPS", "USPS", "Amazon"])
def test_predict_all_carriers(client, predict_payload, carrier):
    payload = {**predict_payload, "carrier": carrier}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["predicted_minutes"] > 0


@pytest.mark.parametrize("route_type", ["urban", "suburban", "rural", "highway"])
def test_predict_all_route_types(client, predict_payload, route_type):
    payload = {**predict_payload, "route_type": route_type}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200


def test_drift_endpoint_returns_200(client):
    resp = client.get("/api/v1/drift")
    assert resp.status_code == 200
    assert "status" in resp.json()
