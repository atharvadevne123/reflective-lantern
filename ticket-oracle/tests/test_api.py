"""API endpoint tests for Ticket-Oracle."""

from __future__ import annotations

import pytest


def test_health_returns_200(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert "model_loaded" in body
    assert "db_ok" in body


def test_health_has_version(client):
    r = client.get("/api/v1/health")
    assert "version" in r.json()


def test_predict_returns_200(client, sample_ticket_payload):
    r = client.post("/api/v1/predict", json=sample_ticket_payload)
    assert r.status_code == 200


def test_predict_response_has_required_fields(client, sample_ticket_payload):
    r = client.post("/api/v1/predict", json=sample_ticket_payload)
    body = r.json()
    for field in ["ticket_id", "priority", "priority_probabilities", "sla_breach_risk",
                  "sla_breach_predicted", "estimated_resolution_hours", "confidence", "model_version"]:
        assert field in body, f"Missing field: {field}"


def test_predict_priority_is_valid_label(client, sample_ticket_payload):
    r = client.post("/api/v1/predict", json=sample_ticket_payload)
    assert r.json()["priority"] in {"P1", "P2", "P3", "P4"}


def test_predict_probabilities_sum_to_one(client, sample_ticket_payload):
    r = client.post("/api/v1/predict", json=sample_ticket_payload)
    probs = r.json()["priority_probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 0.01


def test_predict_sla_breach_risk_in_range(client, sample_ticket_payload):
    r = client.post("/api/v1/predict", json=sample_ticket_payload)
    risk = r.json()["sla_breach_risk"]
    assert 0.0 <= risk <= 1.0


def test_predict_confidence_in_range(client, sample_ticket_payload):
    r = client.post("/api/v1/predict", json=sample_ticket_payload)
    assert 0.0 <= r.json()["confidence"] <= 1.0


def test_predict_estimated_resolution_positive(client, sample_ticket_payload):
    r = client.post("/api/v1/predict", json=sample_ticket_payload)
    assert r.json()["estimated_resolution_hours"] > 0


def test_predict_invalid_channel_returns_422(client, sample_ticket_payload):
    sample_ticket_payload["channel"] = "fax"
    r = client.post("/api/v1/predict", json=sample_ticket_payload)
    assert r.status_code == 422


def test_predict_invalid_tier_returns_422(client, sample_ticket_payload):
    sample_ticket_payload["customer_tier"] = "Platinum"
    r = client.post("/api/v1/predict", json=sample_ticket_payload)
    assert r.status_code == 422


def test_predict_missing_description_returns_422(client, sample_ticket_payload):
    del sample_ticket_payload["description"]
    r = client.post("/api/v1/predict", json=sample_ticket_payload)
    assert r.status_code == 422


def test_predict_short_description_returns_422(client, sample_ticket_payload):
    sample_ticket_payload["description"] = "ok"
    r = client.post("/api/v1/predict", json=sample_ticket_payload)
    assert r.status_code == 422


def test_predict_negative_open_tickets_returns_422(client, sample_ticket_payload):
    sample_ticket_payload["open_tickets_count"] = -1
    r = client.post("/api/v1/predict", json=sample_ticket_payload)
    assert r.status_code == 422


@pytest.mark.parametrize("channel", ["email", "phone", "chat", "portal"])
def test_predict_all_channels(client, sample_ticket_payload, channel):
    sample_ticket_payload["channel"] = channel
    r = client.post("/api/v1/predict", json=sample_ticket_payload)
    assert r.status_code == 200


@pytest.mark.parametrize("tier", ["Bronze", "Standard", "Silver", "Gold"])
def test_predict_all_tiers(client, sample_ticket_payload, tier):
    sample_ticket_payload["customer_tier"] = tier
    r = client.post("/api/v1/predict", json=sample_ticket_payload)
    assert r.status_code == 200


def test_batch_predict_returns_list(client, sample_ticket_payload, sample_p4_payload):
    payload = {"tickets": [sample_ticket_payload, sample_p4_payload]}
    r = client.post("/api/v1/predict/batch", json=payload)
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_batch_predict_empty_list_returns_422(client):
    r = client.post("/api/v1/predict/batch", json={"tickets": []})
    assert r.status_code == 422


def test_metrics_endpoint(client):
    r = client.get("/api/v1/metrics")
    assert r.status_code in {200, 404}


def test_predictions_endpoint_returns_list(client):
    r = client.get("/api/v1/predictions")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_predictions_endpoint_n_param(client):
    r = client.get("/api/v1/predictions?n=5")
    assert r.status_code == 200


def test_drift_endpoint(client):
    r = client.get("/api/v1/drift")
    assert r.status_code == 200


def test_correlation_id_header_present(client, sample_ticket_payload):
    r = client.post("/api/v1/predict", json=sample_ticket_payload)
    assert "x-correlation-id" in {k.lower() for k in r.headers}


def test_correlation_id_propagated(client, sample_ticket_payload):
    cid = "test-correlation-123"
    r = client.post(
        "/api/v1/predict",
        json=sample_ticket_payload,
        headers={"X-Correlation-ID": cid},
    )
    assert r.headers.get("X-Correlation-ID") == cid


def test_ticket_id_in_response(client, sample_ticket_payload):
    r = client.post("/api/v1/predict", json=sample_ticket_payload)
    assert r.json()["ticket_id"] == sample_ticket_payload["ticket_id"]


def test_train_endpoint(client):
    r = client.post("/api/v1/train")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "retrained"
    assert "metrics" in body
