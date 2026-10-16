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


def test_batch_response_has_request_id(client, predict_payload):
    """Batch response includes a request_id field."""
    resp = client.post("/api/v1/predict/batch", json={"shipments": [predict_payload]})
    assert resp.status_code == 200
    assert "request_id" in resp.json()


def test_batch_predictions_have_confidence(client, predict_payload):
    """Each prediction in a batch includes a confidence field between 0 and 1."""
    resp = client.post("/api/v1/predict/batch", json={"shipments": [predict_payload] * 3})
    for pred in resp.json()["predictions"]:
        assert 0.0 <= pred["confidence"] <= 1.0


@pytest.mark.parametrize("carrier", ["DHL", "FedEx", "UPS", "USPS", "Amazon"])
def test_batch_all_carriers_accepted(client, predict_payload, carrier):
    """Batch endpoint accepts all valid carriers."""
    payload = {**predict_payload, "carrier": carrier}
    resp = client.post("/api/v1/predict/batch", json={"shipments": [payload]})
    assert resp.status_code == 200


def test_batch_count_matches_input_length(client, predict_payload):
    """Response count exactly equals the number of shipments submitted."""
    for n in [1, 3, 10]:
        resp = client.post("/api/v1/predict/batch", json={"shipments": [predict_payload] * n})
        assert resp.json()["count"] == n
