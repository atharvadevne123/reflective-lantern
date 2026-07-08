"""API endpoint tests for Temporal-Pulse."""

from __future__ import annotations

import pytest


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_has_status_field(self, client):
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert "status" in data

    def test_health_has_version(self, client):
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert "version" in data
        assert data["version"] == "1.0.0"

    def test_health_has_model_loaded(self, client):
        resp = client.get("/api/v1/health")
        assert "model_loaded" in resp.json()

    def test_health_has_db_connected(self, client):
        resp = client.get("/api/v1/health")
        assert "db_connected" in resp.json()


class TestVersion:
    def test_version_returns_200(self, client):
        resp = client.get("/api/v1/version")
        assert resp.status_code == 200

    def test_version_has_api_version(self, client):
        resp = client.get("/api/v1/version")
        assert "api_version" in resp.json()

    def test_version_has_model_version(self, client):
        resp = client.get("/api/v1/version")
        assert "model_version" in resp.json()


class TestMetrics:
    def test_metrics_returns_200(self, client):
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200

    def test_metrics_has_total_predictions(self, client):
        resp = client.get("/api/v1/metrics")
        assert "total_predictions" in resp.json()


class TestDetect:
    def test_detect_single_reading(self, client, sample_readings):
        payload = {"readings": sample_readings[:5], "horizon": 3}
        resp = client.post("/api/v1/detect", json=payload)
        assert resp.status_code == 200

    def test_detect_returns_results_list(self, client, sample_readings):
        payload = {"readings": sample_readings[:5], "horizon": 3}
        resp = client.post("/api/v1/detect", json=payload)
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 5

    def test_detect_result_has_anomaly_score(self, client, sample_readings):
        payload = {"readings": sample_readings[:3], "horizon": 1}
        resp = client.post("/api/v1/detect", json=payload)
        for result in resp.json()["results"]:
            assert "anomaly_score" in result
            assert 0.0 <= result["anomaly_score"] <= 1.0

    def test_detect_result_has_is_anomaly_bool(self, client, sample_readings):
        payload = {"readings": sample_readings[:3], "horizon": 1}
        resp = client.post("/api/v1/detect", json=payload)
        for result in resp.json()["results"]:
            assert isinstance(result["is_anomaly"], bool)

    def test_detect_empty_readings_rejected(self, client):
        resp = client.post("/api/v1/detect", json={"readings": [], "horizon": 3})
        assert resp.status_code == 422

    def test_detect_invalid_values_rejected(self, client):
        payload = {
            "readings": [
                {
                    "sensor_id": "s1",
                    "timestamp": "2026-01-01T00:00:00",
                    "values": {"temp": float("nan")},
                }
            ],
            "horizon": 1,
        }
        resp = client.post("/api/v1/detect", json=payload)
        assert resp.status_code == 422

    def test_detect_processing_time_returned(self, client, sample_readings):
        payload = {"readings": sample_readings[:5], "horizon": 1}
        resp = client.post("/api/v1/detect", json=payload)
        assert "processing_time_ms" in resp.json()
        assert resp.json()["processing_time_ms"] >= 0

    def test_detect_total_readings_count(self, client, sample_readings):
        n = 10
        payload = {"readings": sample_readings[:n], "horizon": 1}
        resp = client.post("/api/v1/detect", json=payload)
        assert resp.json()["total_readings"] == n


class TestDrift:
    def test_drift_returns_200(self, client):
        resp = client.get("/api/v1/drift")
        assert resp.status_code == 200

    def test_drift_returns_list(self, client):
        resp = client.get("/api/v1/drift")
        assert isinstance(resp.json(), list)


class TestFeatureImportance:
    def test_feature_importance_returns_200(self, client):
        resp = client.get("/api/v1/feature-importance")
        assert resp.status_code == 200

    def test_feature_importance_returns_list(self, client):
        resp = client.get("/api/v1/feature-importance")
        assert isinstance(resp.json(), list)
