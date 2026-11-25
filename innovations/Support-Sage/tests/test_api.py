"""API endpoint tests for Support-Sage."""
import pytest


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_valid_ticket(client, sample_ticket):
    response = client.post("/api/v1/predict", json=sample_ticket)
    assert response.status_code == 200
    data = response.json()
    assert "priority" in data
    assert data["priority"] in ["critical", "high", "medium", "low"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["resolution_hours_estimate"] > 0


def test_predict_critical_ticket(client, critical_ticket):
    response = client.post("/api/v1/predict", json=critical_ticket)
    assert response.status_code == 200
    data = response.json()
    assert "priority" in data
    assert data["priority"] in ["critical", "high", "medium", "low"]


def test_predict_missing_subject(client):
    response = client.post("/api/v1/predict", json={"description": "some description here that is long enough"})
    assert response.status_code == 422


def test_predict_too_short_description(client):
    response = client.post("/api/v1/predict", json={"subject": "Issue", "description": "short"})
    assert response.status_code == 422


def test_metrics_endpoint(client):
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "drift_report" in data


def test_correlation_id_header(client, sample_ticket):
    response = client.post("/api/v1/predict", json=sample_ticket, headers={"X-Correlation-ID": "test-123"})
    assert response.status_code == 200
    assert response.headers.get("X-Correlation-ID") == "test-123"


def test_response_time_header(client):
    response = client.get("/api/v1/health")
    assert "X-Response-Time-Ms" in response.headers


@pytest.mark.parametrize("ticket", [
    {"subject": "Password reset request for user john.doe", "description": "User john.doe has forgotten their password and needs a reset link sent to their corporate email address."},
    {"subject": "New software installation request", "description": "Please install Adobe Acrobat Pro on my laptop. License has been approved by my manager. Asset tag: LT-2024-99."},
    {"subject": "Internet connectivity issues in conference room B", "description": "The wifi in conference room B is very slow and dropping packets. Affecting all meetings scheduled there today."},
])
def test_predict_various_tickets(client, ticket):
    response = client.post("/api/v1/predict", json=ticket)
    assert response.status_code == 200
    assert response.json()["priority"] in ["critical", "high", "medium", "low"]
