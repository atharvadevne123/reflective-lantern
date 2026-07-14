"""Tests for FastAPI endpoints."""

from __future__ import annotations

import pytest


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_has_model_loaded_flag(self, client):
        resp = client.get("/api/v1/health")
        assert "model_loaded" in resp.json()


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200

    def test_metrics_has_expected_keys(self, client):
        data = client.get("/api/v1/metrics").json()
        assert "model_metrics" in data
        assert "prediction_stats" in data


class TestPredictEndpoint:
    def test_predict_single_reading(self, client, sample_reading):
        payload = {"readings": [sample_reading], "horizon_h": 1}
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions_kwh"]) == 1
        assert data["predictions_kwh"][0] >= 0
        assert data["horizon_h"] == 1
        assert data["meter_ids"] == ["meter_001"]

    def test_predict_multiple_readings(self, client, sample_readings):
        payload = {"readings": sample_readings, "horizon_h": 1}
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200
        assert len(resp.json()["predictions_kwh"]) == 3

    def test_predict_horizon_multiplies_output(self, client, sample_reading):
        r1 = client.post("/api/v1/predict", json={"readings": [sample_reading], "horizon_h": 1}).json()
        r6 = client.post("/api/v1/predict", json={"readings": [sample_reading], "horizon_h": 6}).json()
        assert r6["predictions_kwh"][0] > r1["predictions_kwh"][0]

    def test_predict_invalid_building_type(self, client, sample_reading):
        bad = {**sample_reading, "building_type": "spaceship"}
        resp = client.post("/api/v1/predict", json={"readings": [bad], "horizon_h": 1})
        assert resp.status_code == 422

    def test_predict_negative_consumption_rejected(self, client, sample_reading):
        bad = {**sample_reading, "consumption_kwh": -5.0}
        resp = client.post("/api/v1/predict", json={"readings": [bad], "horizon_h": 1})
        assert resp.status_code == 422

    def test_predict_empty_readings_rejected(self, client):
        resp = client.post("/api/v1/predict", json={"readings": [], "horizon_h": 1})
        assert resp.status_code == 422

    @pytest.mark.parametrize("hour", [0, 6, 12, 18, 23])
    def test_predict_various_hours(self, client, sample_reading, hour):
        reading = {**sample_reading, "hour_of_day": hour}
        resp = client.post("/api/v1/predict", json={"readings": [reading], "horizon_h": 1})
        assert resp.status_code == 200
        assert resp.json()["predictions_kwh"][0] >= 0

    @pytest.mark.parametrize("building", ["residential", "commercial", "industrial", "office"])
    def test_predict_building_types(self, client, sample_reading, building):
        reading = {**sample_reading, "building_type": building}
        resp = client.post("/api/v1/predict", json={"readings": [reading], "horizon_h": 1})
        assert resp.status_code == 200


class TestAnomalyEndpoint:
    def test_anomaly_normal_consumption(self, client, sample_reading):
        payload = {"meter_id": "meter_001", "consumption_kwh": 4.5}
        resp = client.post("/api/v1/anomaly", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "is_anomaly" in data
        assert "anomaly_score" in data
        assert "severity" in data

    def test_anomaly_high_consumption_flagged(self, client):
        payload = {"meter_id": "meter_001", "consumption_kwh": 9999.0}
        resp = client.post("/api/v1/anomaly", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_anomaly"] is True

    def test_anomaly_returns_anomaly_type(self, client):
        resp = client.post("/api/v1/anomaly", json={"meter_id": "m1", "consumption_kwh": 5.0})
        assert "anomaly_type" in resp.json()

    @pytest.mark.parametrize("kwh", [0.1, 2.0, 5.0, 10.0, 20.0])
    def test_anomaly_various_consumption_levels(self, client, kwh):
        resp = client.post("/api/v1/anomaly", json={"meter_id": "m1", "consumption_kwh": kwh})
        assert resp.status_code == 200


class TestDriftEndpoint:
    def test_drift_no_drift(self, client):
        payload = {"feature_values": {"consumption_kwh": [3.0, 4.0, 3.5, 4.2, 3.8, 3.9, 4.1]}}
        resp = client.post("/api/v1/drift", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "feature_results" in data
        assert "drift_detected" in data

    def test_drift_detects_shift(self, client):
        payload = {"feature_values": {"consumption_kwh": [100.0] * 20}}
        resp = client.post("/api/v1/drift", json=payload)
        assert resp.status_code == 200

    def test_drift_multiple_features(self, client):
        payload = {
            "feature_values": {
                "consumption_kwh": [3.0, 4.0, 3.5, 4.2, 3.8],
                "temperature_c": [20.0, 21.0, 22.0, 19.0, 20.5],
            }
        }
        resp = client.post("/api/v1/drift", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "total_features_checked" in data
        assert data["total_features_checked"] == 2


class TestForecastEndpoint:
    def test_forecast_returns_steps(self, client, sample_reading):
        payload = {"readings": [sample_reading], "steps": 6}
        resp = client.post("/api/v1/forecast", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["steps"] == 6
        assert len(data["forecast"]) == 6

    def test_forecast_24h(self, client, sample_reading):
        resp = client.post("/api/v1/forecast", json={"readings": [sample_reading], "steps": 24})
        assert resp.status_code == 200
        assert len(resp.json()["forecast"]) == 24


class TestBatchPredictEndpoint:
    def test_batch_predict(self, client, sample_readings):
        payload = {"readings": sample_readings, "horizon_h": 1}
        resp = client.post("/api/v1/batch-predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3
        assert len(data["results"]) == 3
