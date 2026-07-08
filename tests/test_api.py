"""Tests for FastAPI endpoints."""

import pytest


def test_health_returns_ok(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_version" in data
    assert "db_connected" in data


def test_metrics_endpoint(client):
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_version" in data


def test_predict_returns_valuation(client, sample_property):
    resp = client.post("/api/v1/predict", json=sample_property)
    assert resp.status_code == 200
    data = resp.json()
    assert data["predicted_value"] > 0
    assert 0 <= data["investment_score"] <= 10
    assert data["confidence_band_low"] < data["predicted_value"]
    assert data["confidence_band_high"] > data["predicted_value"]
    assert "correlation_id" in data
    assert "model_version" in data


def test_predict_confidence_band_width(client, sample_property):
    resp = client.post("/api/v1/predict", json=sample_property)
    data = resp.json()
    low = data["confidence_band_low"]
    high = data["confidence_band_high"]
    mid = data["predicted_value"]
    assert abs(low - mid * 0.92) < 1
    assert abs(high - mid * 1.08) < 1


@pytest.mark.parametrize(
    "sqft,beds,baths",
    [
        (800, 1, 1.0),
        (1500, 2, 1.5),
        (3000, 4, 3.0),
        (4500, 5, 4.0),
    ],
)
def test_predict_various_sizes(client, sample_property, sqft, beds, baths):
    prop = dict(sample_property)
    prop["sqft"] = sqft
    prop["bedrooms"] = beds
    prop["bathrooms"] = baths
    resp = client.post("/api/v1/predict", json=prop)
    assert resp.status_code == 200
    assert resp.json()["predicted_value"] > 0


def test_predict_missing_required_field(client, sample_property):
    prop = dict(sample_property)
    del prop["sqft"]
    resp = client.post("/api/v1/predict", json=prop)
    assert resp.status_code == 422


def test_predict_invalid_sqft(client, sample_property):
    prop = dict(sample_property)
    prop["sqft"] = -100
    resp = client.post("/api/v1/predict", json=prop)
    assert resp.status_code == 422


def test_predict_invalid_year_built(client, sample_property):
    prop = dict(sample_property)
    prop["year_built"] = 1700
    resp = client.post("/api/v1/predict", json=prop)
    assert resp.status_code == 422


def test_batch_predict(client, sample_property):
    body = {"properties": [sample_property, sample_property]}
    resp = client.post("/api/v1/batch-predict", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2


def test_batch_predict_empty_raises(client):
    resp = client.post("/api/v1/batch-predict", json={"properties": []})
    assert resp.status_code == 422


def test_comparable_properties(client, sample_property):
    body = {"property": sample_property, "top_k": 3}
    resp = client.post("/api/v1/comparable-properties", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert "comparables" in data
    assert "query_vector_dim" in data


def test_neighborhood_stats_returns_default(client):
    resp = client.get("/api/v1/neighborhood-stats/99999")
    assert resp.status_code == 200
    data = resp.json()
    assert data["zipcode"] == "99999"
    assert data["median_price"] > 0


def test_drift_status(client):
    resp = client.get("/api/v1/drift-status")
    assert resp.status_code == 200
    data = resp.json()
    assert "drift_reports" in data
    assert "total_predictions" in data


def test_run_drift_check_insufficient_data(client):
    resp = client.post("/api/v1/run-drift-check")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "skipped"


def test_correlation_id_header(client, sample_property):
    resp = client.post(
        "/api/v1/predict",
        json=sample_property,
        headers={"X-Correlation-ID": "test-corr-123"},
    )
    assert resp.headers.get("x-correlation-id") == "test-corr-123"


def test_response_time_header(client):
    resp = client.get("/api/v1/health")
    assert "x-response-time-ms" in resp.headers


def test_neighborhood_stats_known_zip(client, db_session):
    from app.database import NeighborhoodStat

    stat = NeighborhoodStat(
        zipcode="55555",
        median_price=400_000.0,
        median_price_per_sqft=250.0,
        school_score=7.0,
        transit_score=6.5,
        walkability_score=6.0,
        crime_rate=0.35,
        avg_rental_yield=0.07,
    )
    db_session.add(stat)
    db_session.commit()


def test_predict_stores_prediction_in_db(client, db_session, sample_property):
    from app.database import PredictionLog

    before = db_session.query(PredictionLog).count()
    client.post("/api/v1/predict", json=sample_property)
    after = db_session.query(PredictionLog).count()
    assert after >= before


def test_batch_predict_all_positive(client, sample_property):
    body = {"properties": [sample_property] * 3}
    resp = client.post("/api/v1/batch-predict", json=body)
    data = resp.json()
    assert all(p["predicted_value"] > 0 for p in data["predictions"])
