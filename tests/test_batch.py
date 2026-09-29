"""Tests for the bulk prediction endpoint."""
from __future__ import annotations

import pytest


def test_batch_returns_all_predictions(client, predict_payload):
    body = {"shipments": [predict_payload, predict_payload, predict_payload]}
    resp = client.post("/api/v1/predict/batch", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 3
    assert len(data["predictions"]) == 3


def test_batch_predictions_are_positive(client, predict_payload):
    resp = client.post("/api/v1/predict/batch", json={"shipments": [predict_payload]})
    assert all(p["predicted_minutes"] > 0 for p in resp.json()["predictions"])


def test_batch_rejects_empty_list(client):
    resp = client.post("/api/v1/predict/batch", json={"shipments": []})
    assert resp.status_code == 422


def test_batch_rejects_over_100(client, predict_payload):
    resp = client.post("/api/v1/predict/batch", json={"shipments": [predict_payload] * 101})
    assert resp.status_code == 422


def test_batch_rejects_invalid_member(client, predict_payload):
    bad = {**predict_payload, "carrier": "Nope"}
    resp = client.post("/api/v1/predict/batch", json={"shipments": [predict_payload, bad]})
    assert resp.status_code == 422


@pytest.mark.parametrize("n", [1, 5, 20])
def test_batch_sizes(client, predict_payload, n):
    resp = client.post("/api/v1/predict/batch", json={"shipments": [predict_payload] * n})
    assert resp.json()["count"] == n


def test_batch_response_has_model_version(client, predict_payload):
    resp = client.post("/api/v1/predict/batch", json={"shipments": [predict_payload]})
    predictions = resp.json()["predictions"]
    assert all("model_version" in p for p in predictions)


def test_batch_response_has_confidence(client, predict_payload):
    resp = client.post("/api/v1/predict/batch", json={"shipments": [predict_payload]})
    predictions = resp.json()["predictions"]
    assert all(0.0 <= p["confidence"] <= 1.0 for p in predictions)


def test_batch_response_hours_consistent(client, predict_payload):
    """predicted_hours must equal predicted_minutes / 60 within floating-point tolerance."""
    resp = client.post("/api/v1/predict/batch", json={"shipments": [predict_payload]})
    for pred in resp.json()["predictions"]:
        assert abs(pred["predicted_hours"] - pred["predicted_minutes"] / 60) < 0.01


def test_batch_mixed_carriers(client, predict_payload):
    carriers = ["DHL", "FedEx", "UPS", "USPS", "Amazon"]
    shipments = [{**predict_payload, "carrier": c} for c in carriers]
    resp = client.post("/api/v1/predict/batch", json={"shipments": shipments})
    assert resp.status_code == 200
    assert resp.json()["count"] == 5


def test_batch_exactly_100_allowed(client, predict_payload):
    resp = client.post("/api/v1/predict/batch", json={"shipments": [predict_payload] * 100})
    assert resp.status_code == 200
    assert resp.json()["count"] == 100
