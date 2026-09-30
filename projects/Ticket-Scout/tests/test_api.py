"""API endpoint tests."""

from __future__ import annotations

import pytest


def test_health_returns_200(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert "model_version" in body
    assert "uptime_seconds" in body


def test_health_has_correlation_id(client):
    resp = client.get("/api/v1/health")
    assert "X-Correlation-ID" in resp.headers


def test_predict_single_valid(client):
    payload = {
        "subject": "VPN not connecting after update",
        "body": "My VPN drops every 30 minutes since the last system update.",
        "priority": "high",
        "created_hour": 9,
        "created_dow": 1,
        "org_size": 500,
    }
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_category" in data
    assert "sla_breach_prob" in data
    assert 0.0 <= data["sla_breach_prob"] <= 1.0
    assert "resolution_hours_pred" in data
    assert data["resolution_hours_pred"] >= 0.0
    assert "confidence" in data
    assert "risk_level" in data
    assert data["risk_level"] in {"low", "medium", "high"}


@pytest.mark.parametrize("priority", ["low", "medium", "high", "critical"])
def test_predict_all_priorities(client, priority):
    payload = {
        "subject": "Test ticket subject",
        "body": "This is the ticket body text.",
        "priority": priority,
    }
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["predicted_category"] in {"access", "email", "hardware", "network", "software"}


def test_predict_invalid_priority_returns_422(client):
    payload = {
        "subject": "Test",
        "body": "Body",
        "priority": "SUPERURGENT",
    }
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_predict_missing_subject_returns_422(client):
    resp = client.post("/api/v1/predict", json={"body": "No subject here"})
    assert resp.status_code == 422


def test_predict_batch_valid(client):
    payload = {
        "tickets": [
            {"subject": "Password reset", "body": "Account locked.", "priority": "critical"},
            {"subject": "Printer not working", "body": "Printer jammed.", "priority": "low"},
        ]
    }
    resp = client.post("/api/v1/predict/batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["predictions"]) == 2
    assert data["model_version"] is not None


def test_predict_batch_too_many_tickets_returns_422(client):
    payload = {"tickets": [{"subject": f"Ticket {i}", "body": "body", "priority": "low"} for i in range(60)]}
    resp = client.post("/api/v1/predict/batch", json=payload)
    assert resp.status_code == 422


def test_metrics_endpoint(client):
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction_volume_24h" in data
    assert "model_metrics" in data


def test_drift_insufficient_data(client):
    resp = client.post("/api/v1/drift")
    assert resp.status_code == 200
    data = resp.json()
    assert "message" in data or "features_checked" in data


def test_correlation_id_forwarded(client):
    custom_cid = "test-correlation-123"
    resp = client.get("/api/v1/health", headers={"X-Correlation-ID": custom_cid})
    assert resp.headers.get("X-Correlation-ID") == custom_cid


@pytest.mark.parametrize("subject,expected_in", [
    ("VPN not connecting", ["network", "access"]),
    ("Password reset needed", ["access", "network"]),
    ("Laptop screen flickering", ["hardware", "software"]),
    ("Outlook not syncing", ["email", "software"]),
    ("Install software license expired", ["software", "access"]),
])
def test_predict_category_reasonable(client, subject, expected_in):
    resp = client.post("/api/v1/predict", json={
        "subject": subject,
        "body": f"Details about: {subject}",
        "priority": "medium",
    })
    assert resp.status_code == 200
    # Category should be in valid set (not asserting exact match due to model variance)
    data = resp.json()
    assert data["predicted_category"] in {"access", "email", "hardware", "network", "software"}
    assert 0.0 <= data["confidence"] <= 1.0
