"""API endpoint tests for Fleet-Mind."""

from __future__ import annotations

import pytest


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_expected_keys(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_version" in data
        assert "db_ok" in data

    def test_health_db_ok(self, client):
        data = client.get("/health").json()
        assert data["db_ok"] is True


class TestVersionEndpoint:
    """Tests for GET /api/v1/version."""

    def test_version_200(self, client):
        assert client.get("/api/v1/version").status_code == 200

    def test_version_fields(self, client):
        data = client.get("/api/v1/version").json()
        assert "api_version" in data
        assert "model_version" in data


class TestPredictEndpoint:
    """Tests for POST /api/v1/predict."""

    def test_predict_normal_reading(self, client, sample_reading):
        resp = client.post("/api/v1/predict", json=sample_reading)
        assert resp.status_code == 200
        data = resp.json()
        assert "maintenance_needed" in data
        assert 0.0 <= data["anomaly_score"] <= 1.0
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_anomalous_reading(self, client, anomalous_reading):
        resp = client.post("/api/v1/predict", json=anomalous_reading)
        assert resp.status_code == 200
        data = resp.json()
        assert data["anomaly_type"] is not None

    def test_predict_invalid_speed(self, client, sample_reading):
        bad = {**sample_reading, "speed_kmh": 9999}
        resp = client.post("/api/v1/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_missing_field(self, client, sample_reading):
        bad = {k: v for k, v in sample_reading.items() if k != "rpm"}
        resp = client.post("/api/v1/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_vehicle_id_in_response(self, client, sample_reading):
        resp = client.post("/api/v1/predict", json=sample_reading)
        assert resp.json()["vehicle_id"] == sample_reading["vehicle_id"]

    @pytest.mark.parametrize("engine_temp", [60.0, 90.0, 120.0])
    def test_predict_varying_engine_temp(self, client, sample_reading, engine_temp):
        reading = {**sample_reading, "engine_temp_c": engine_temp}
        resp = client.post("/api/v1/predict", json=reading)
        assert resp.status_code == 200


class TestBatchPredictEndpoint:
    """Tests for POST /api/v1/predict/batch."""

    def test_batch_predict(self, client, sample_reading, anomalous_reading):
        payload = {"readings": [sample_reading, anomalous_reading]}
        resp = client.post("/api/v1/predict/batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert "maintenance_flagged" in data

    def test_batch_empty_raises_422(self, client):
        resp = client.post("/api/v1/predict/batch", json={"readings": []})
        assert resp.status_code == 422

    def test_batch_results_count(self, client, sample_reading):
        readings = [sample_reading] * 5
        resp = client.post("/api/v1/predict/batch", json={"readings": readings})
        assert resp.json()["total"] == 5


class TestMetricsEndpoint:
    """Tests for GET /api/v1/metrics."""

    def test_metrics_200(self, client):
        assert client.get("/api/v1/metrics").status_code == 200

    def test_metrics_maintenance_rate_present(self, client):
        data = client.get("/api/v1/metrics").json()
        assert "maintenance_rate" in data


class TestDriftEndpoint:
    """Tests for GET /api/v1/drift."""

    def test_drift_200(self, client):
        assert client.get("/api/v1/drift").status_code == 200

    def test_drift_has_fields(self, client):
        data = client.get("/api/v1/drift").json()
        assert "checked_features" in data
        assert "drifted_features" in data
        assert "reports" in data
