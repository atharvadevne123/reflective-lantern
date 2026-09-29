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


def test_predict_confidence_range(client, predict_payload):
    resp = client.post("/api/v1/predict", json=predict_payload)
    data = resp.json()
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_request_id_header(client, predict_payload):
    resp = client.post("/api/v1/predict", json=predict_payload)
    assert "x-request-id" in resp.headers


def test_predict_response_time_header(client, predict_payload):
    resp = client.post("/api/v1/predict", json=predict_payload)
    assert "x-response-time-ms" in resp.headers
    assert float(resp.headers["x-response-time-ms"]) >= 0


def test_predict_custom_request_id_echoed(client, predict_payload):
    custom_id = "test-abc-123"
    resp = client.post(
        "/api/v1/predict",
        json=predict_payload,
        headers={"X-Request-ID": custom_id},
    )
    assert resp.headers.get("x-request-id") == custom_id


def test_batch_predict_single_item(client, predict_payload):
    resp = client.post("/api/v1/predict/batch", json={"shipments": [predict_payload]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["predicted_minutes"] > 0


def test_batch_predict_multiple_items(client, predict_payload):
    shipments = [predict_payload] * 3
    resp = client.post("/api/v1/predict/batch", json={"shipments": shipments})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 3
    assert len(data["predictions"]) == 3


def test_batch_predict_empty_list_rejected(client):
    resp = client.post("/api/v1/predict/batch", json={"shipments": []})
    assert resp.status_code == 422


def test_health_model_loaded_field(client):
    resp = client.get("/api/v1/health")
    data = resp.json()
    assert isinstance(data["model_loaded"], bool)


def test_metrics_fields_present(client):
    resp = client.get("/api/v1/metrics")
    data = resp.json()
    assert "model_version" in data
    assert "rmse_mean" in data
    assert "r2_mean" in data
