"""API endpoint tests for Sports-Oracle."""

from __future__ import annotations

import pytest


def test_health_ok(api_client):
    r = api_client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "model_loaded" in body


def test_predict_valid(api_client, sample_match_payload):
    r = api_client.post("/api/v1/predict", json=sample_match_payload)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_outcome"] in ("H", "D", "A")
    assert 0.0 <= body["prob_home"] <= 1.0
    assert 0.0 <= body["prob_draw"] <= 1.0
    assert 0.0 <= body["prob_away"] <= 1.0
    prob_sum = body["prob_home"] + body["prob_draw"] + body["prob_away"]
    assert abs(prob_sum - 1.0) < 0.01


def test_predict_returns_request_id(api_client, sample_match_payload):
    r = api_client.post("/api/v1/predict", json=sample_match_payload)
    assert r.status_code == 200
    assert "request_id" in r.json()


@pytest.mark.parametrize(
    "bad_field,bad_value",
    [
        ("home_form", 1.5),
        ("away_form", -0.1),
        ("home_rest_days", 0),
        ("away_rest_days", 50),
    ],
)
def test_predict_invalid_input(api_client, sample_match_payload, bad_field, bad_value):
    payload = {**sample_match_payload, bad_field: bad_value}
    r = api_client.post("/api/v1/predict", json=payload)
    assert r.status_code == 422


def test_predict_missing_required_field(api_client, sample_match_payload):
    payload = {k: v for k, v in sample_match_payload.items() if k != "home_form"}
    r = api_client.post("/api/v1/predict", json=payload)
    assert r.status_code == 422


def test_predict_batch_valid(api_client, sample_match_payload):
    r = api_client.post(
        "/api/v1/predict/batch", json={"matches": [sample_match_payload, sample_match_payload]}
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    for item in body:
        assert item["predicted_outcome"] in ("H", "D", "A")


def test_predict_batch_empty(api_client):
    r = api_client.post("/api/v1/predict/batch", json={"matches": []})
    assert r.status_code == 422


def test_metrics_endpoint(api_client):
    r = api_client.get("/api/v1/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "auc_mean" in body
    assert "accuracy_mean" in body


def test_drift_endpoint(api_client):
    reference = [{"home_form": 0.7, "away_form": 0.4} for _ in range(50)]
    current = [{"home_form": 0.6, "away_form": 0.5} for _ in range(30)]
    r = api_client.post(
        "/api/v1/drift", json={"reference_features": reference, "current_features": current}
    )
    assert r.status_code == 200
    body = r.json()
    assert "drift_detected" in body
    assert "features_checked" in body


def test_drift_endpoint_insufficient_data(api_client):
    reference = [{"home_form": 0.7} for _ in range(3)]
    current = [{"home_form": 0.6} for _ in range(3)]
    r = api_client.post(
        "/api/v1/drift", json={"reference_features": reference, "current_features": current}
    )
    assert r.status_code == 200
    body = r.json()
    # With insufficient samples, all features should show no drift
    for v in body["details"].values():
        assert not v["drift_detected"]


def test_correlation_id_header_returned(api_client, sample_match_payload):
    r = api_client.post(
        "/api/v1/predict",
        json=sample_match_payload,
        headers={"X-Correlation-ID": "test-abc-123"},
    )
    assert r.headers.get("X-Correlation-ID") == "test-abc-123"
