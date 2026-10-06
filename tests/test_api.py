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


def test_health_alias_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()


def test_metrics_alias_returns_200(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "model_version" in resp.json()


def test_version_alias_returns_200(client):
    resp = client.get("/version")
    assert resp.status_code == 200
    assert "version" in resp.json()


def test_predict_batch_valid(client, predict_payload):
    resp = client.post("/api/v1/predict/batch", json={"shipments": [predict_payload]})
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions" in data
    assert data["count"] == 1
    assert data["predictions"][0]["predicted_minutes"] > 0


def test_predict_batch_multiple_shipments(client, predict_payload):
    resp = client.post(
        "/api/v1/predict/batch", json={"shipments": [predict_payload, predict_payload]}
    )
    assert resp.status_code == 200
    assert resp.json()["count"] == 2


def test_predict_confidence_in_range(client, predict_payload):
    resp = client.post("/api/v1/predict", json=predict_payload)
    assert resp.status_code == 200
    confidence = resp.json()["confidence"]
    assert 0.0 <= confidence <= 1.0


@pytest.mark.parametrize(
    "carrier",
    ["DHL", "FedEx", "UPS", "USPS", "Amazon"],
)
def test_predict_all_valid_carriers(client, predict_payload, carrier: str) -> None:
    payload = {**predict_payload, "carrier": carrier}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200, f"Carrier {carrier} returned {resp.status_code}"
    assert resp.json()["predicted_minutes"] > 0


@pytest.mark.parametrize(
    "route_type",
    ["urban", "suburban", "rural", "highway"],
)
def test_predict_all_route_types(client, predict_payload, route_type: str) -> None:
    payload = {**predict_payload, "route_type": route_type}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200, f"Route {route_type} returned {resp.status_code}"


@pytest.mark.parametrize(
    "field,bad_value,expected_status",
    [
        ("carrier", "InvalidCarrier", 422),
        ("route_type", "underwater", 422),
        ("distance_km", -10.0, 422),
        ("weight_kg", 0.0, 422),
    ],
)
def test_predict_invalid_inputs_rejected(
    client, predict_payload, field: str, bad_value: object, expected_status: int
) -> None:
    payload = {**predict_payload, field: bad_value}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == expected_status, (
        f"Expected {expected_status} for {field}={bad_value!r}"
    )
