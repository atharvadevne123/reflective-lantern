"""API endpoint tests for Property-Sage."""

import pytest


def test_health_returns_ok(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "property-sage"


def test_predict_returns_valid_response(client, sample_property):
    response = client.post("/api/v1/predict", json=sample_property)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert "predicted_rental_yield" in data
    assert "estimated_annual_rental" in data
    assert "estimated_monthly_rental" in data
    assert "request_id" in data
    assert data["predicted_price"] > 0
    assert 0 < data["predicted_rental_yield"] < 1


def test_predict_requires_all_fields(client):
    response = client.post("/api/v1/predict", json={"bedrooms": 3})
    assert response.status_code == 422


def test_predict_rejects_invalid_neighborhood(client, sample_property):
    body = {**sample_property, "neighborhood": "moon_base"}
    response = client.post("/api/v1/predict", json=body)
    assert response.status_code == 422


def test_predict_rejects_invalid_property_type(client, sample_property):
    body = {**sample_property, "property_type": "spaceship"}
    response = client.post("/api/v1/predict", json=body)
    assert response.status_code == 422


def test_predict_accepts_optional_lot_size(client, sample_property):
    body = {k: v for k, v in sample_property.items() if k != "lot_size"}
    response = client.post("/api/v1/predict", json=body)
    assert response.status_code == 200


def test_metrics_endpoint(client):
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "prediction_stats" in data


def test_drift_check_endpoint(client):
    response = client.post("/api/v1/drift-check")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_root_returns_message(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


@pytest.mark.parametrize("neighborhood", ["downtown", "waterfront", "rural"])
def test_predict_all_neighborhoods(client, sample_property, neighborhood):
    body = {**sample_property, "neighborhood": neighborhood}
    response = client.post("/api/v1/predict", json=body)
    assert response.status_code == 200
    assert response.json()["predicted_price"] > 0


@pytest.mark.parametrize("property_type", ["apartment", "house", "condo", "townhouse", "studio"])
def test_predict_all_property_types(client, sample_property, property_type):
    body = {**sample_property, "property_type": property_type}
    response = client.post("/api/v1/predict", json=body)
    assert response.status_code == 200


def test_correlation_id_in_response_header(client):
    response = client.get("/api/v1/health")
    assert "x-correlation-id" in response.headers


def test_response_time_header_present(client):
    response = client.get("/api/v1/health")
    assert "x-response-time-ms" in response.headers
