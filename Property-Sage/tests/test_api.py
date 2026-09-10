"""API endpoint tests for Property-Sage — extended parametrized suite."""

import pytest


def test_health_returns_ok(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "property-sage"
    assert "version" in data


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


def test_predict_rejects_zero_bedrooms(client, sample_property):
    body = {**sample_property, "bedrooms": 0}
    response = client.post("/api/v1/predict", json=body)
    assert response.status_code == 422


def test_predict_rejects_future_year_built(client, sample_property):
    body = {**sample_property, "year_built": 2099}
    response = client.post("/api/v1/predict", json=body)
    assert response.status_code == 422


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


def test_neighbourhood_report_endpoint(client):
    response = client.get("/api/v1/neighbourhood/suburb")
    assert response.status_code == 200
    data = response.json()
    assert "market_report" in data
    assert len(data["market_report"]) > 10


def test_neighbourhood_search_endpoint(client):
    response = client.get("/api/v1/neighbourhood-search", params={"q": "high rental yield"})
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)


def test_correlation_id_in_response_header(client):
    response = client.get("/api/v1/health")
    assert "x-correlation-id" in response.headers


def test_response_time_header_present(client):
    response = client.get("/api/v1/health")
    assert "x-response-time-ms" in response.headers


def test_custom_correlation_id_echoed(client):
    response = client.get("/api/v1/health", headers={"X-Correlation-ID": "test-id-abc"})
    assert response.headers.get("x-correlation-id") == "test-id-abc"


@pytest.mark.parametrize("neighborhood", ["downtown", "waterfront", "rural", "university", "airport"])
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


@pytest.mark.parametrize("year_built,expected_status", [
    (2020, 200),
    (1900, 200),
    (1799, 422),
    (2025, 422),
])
def test_year_built_boundary(client, sample_property, year_built, expected_status):
    body = {**sample_property, "year_built": year_built}
    response = client.post("/api/v1/predict", json=body)
    assert response.status_code == expected_status


def test_annual_equals_12x_monthly(client, sample_property):
    response = client.post("/api/v1/predict", json=sample_property)
    data = response.json()
    assert abs(data["estimated_annual_rental"] - data["estimated_monthly_rental"] * 12) < 1.0
