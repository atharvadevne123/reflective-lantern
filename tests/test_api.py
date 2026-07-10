"""Tests for Volt-Cast API endpoints."""

from __future__ import annotations

import pytest


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_response_schema(self, client):
        data = client.get("/api/v1/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "prediction_count" in data
        assert "uptime_seconds" in data

    def test_health_model_is_loaded(self, client):
        data = client.get("/api/v1/health").json()
        assert data["model_loaded"] is True
        assert data["status"] == "ok"


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200

    def test_metrics_has_model_version(self, client):
        data = client.get("/api/v1/metrics").json()
        assert "model_version" in data
        assert "prediction_count" in data


class TestPredictEndpoint:
    def test_predict_returns_200(self, client, sample_predict_payload):
        resp = client.post("/api/v1/predict", json=sample_predict_payload)
        assert resp.status_code == 200

    def test_predict_response_has_load(self, client, sample_predict_payload):
        data = client.post("/api/v1/predict", json=sample_predict_payload).json()
        assert "predicted_load_mw" in data
        assert isinstance(data["predicted_load_mw"], float)
        assert data["predicted_load_mw"] > 0

    def test_predict_load_in_plausible_range(self, client, sample_predict_payload):
        data = client.post("/api/v1/predict", json=sample_predict_payload).json()
        assert 500.0 <= data["predicted_load_mw"] <= 15000.0

    def test_predict_returns_request_id(self, client, sample_predict_payload):
        data = client.post("/api/v1/predict", json=sample_predict_payload).json()
        assert "request_id" in data
        assert len(data["request_id"]) > 0

    def test_predict_returns_latency(self, client, sample_predict_payload):
        data = client.post("/api/v1/predict", json=sample_predict_payload).json()
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0

    @pytest.mark.parametrize("hour", [0, 6, 12, 18, 23])
    def test_predict_various_hours(self, client, sample_predict_payload, hour):
        payload = {**sample_predict_payload, "hour": hour}
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200
        assert resp.json()["predicted_load_mw"] > 0

    @pytest.mark.parametrize("month", [1, 3, 6, 9, 12])
    def test_predict_various_months(self, client, sample_predict_payload, month):
        payload = {**sample_predict_payload, "month": month}
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200

    def test_predict_invalid_hour_fails(self, client, sample_predict_payload):
        payload = {**sample_predict_payload, "hour": 25}
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_invalid_temp_fails(self, client, sample_predict_payload):
        payload = {**sample_predict_payload, "temperature_c": 999.0}
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_invalid_humidity_fails(self, client, sample_predict_payload):
        payload = {**sample_predict_payload, "humidity_pct": -5.0}
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_without_history(self, client):
        payload = {
            "hour": 10,
            "day_of_week": 1,
            "month": 5,
            "is_weekend": False,
            "temperature_c": 20.0,
            "humidity_pct": 50.0,
        }
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200

    def test_predict_weekend(self, client, sample_predict_payload):
        payload = {**sample_predict_payload, "is_weekend": True}
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200


class TestBatchPredictEndpoint:
    def test_batch_predict_single(self, client, sample_predict_payload):
        resp = client.post("/api/v1/batch-predict", json={"requests": [sample_predict_payload]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert len(data["predictions"]) == 1

    def test_batch_predict_multiple(self, client, sample_predict_payload):
        payloads = [
            {**sample_predict_payload, "hour": h} for h in [8, 12, 18]
        ]
        resp = client.post("/api/v1/batch-predict", json={"requests": payloads})
        assert resp.status_code == 200
        assert resp.json()["total"] == 3

    def test_batch_predict_empty_fails(self, client):
        resp = client.post("/api/v1/batch-predict", json={"requests": []})
        assert resp.status_code == 422


class TestDriftEndpoint:
    def test_drift_returns_200(self, client):
        resp = client.get("/api/v1/drift")
        assert resp.status_code == 200

    def test_drift_response_schema(self, client):
        data = client.get("/api/v1/drift").json()
        assert "drifts" in data
        assert "summary" in data
        assert "model_version" in data


class TestForecastEndpoint:
    def test_forecast_returns_200(self, client):
        resp = client.get(
            "/api/v1/forecast",
            params={"start_hour": 8, "day_of_week": 0, "month": 6, "horizon_hours": 12}
        )
        assert resp.status_code == 200

    def test_forecast_returns_correct_count(self, client):
        resp = client.get(
            "/api/v1/forecast",
            params={"start_hour": 0, "day_of_week": 1, "month": 3, "horizon_hours": 24}
        )
        data = resp.json()
        assert len(data["forecasts"]) == 24
        assert data["horizon_hours"] == 24

    def test_forecast_hour_wraps_correctly(self, client):
        resp = client.get(
            "/api/v1/forecast",
            params={"start_hour": 22, "day_of_week": 3, "month": 8, "horizon_hours": 6}
        )
        data = resp.json()
        hours = [f["hour"] for f in data["forecasts"]]
        assert hours[0] == 22
        assert hours[2] == 0  # wraps past midnight


class TestRetrainEndpoint:
    def test_retrain_returns_200(self, client):
        resp = client.post("/api/v1/retrain")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "retrained"
        assert "metrics" in data
