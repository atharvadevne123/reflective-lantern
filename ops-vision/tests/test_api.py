"""Tests for Ops-Vision API endpoints."""

import pytest


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_root_health_returns_ok(self, client):
        """Root health check should return status ok."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_root_health_returns_version(self, client):
        """Root health check should include a version field."""
        resp = client.get("/health")
        assert "version" in resp.json()

    def test_v1_health_returns_ok(self, client):
        """API v1 health check should return status ok or degraded."""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] in ("ok", "degraded")

    def test_v1_health_has_model_loaded_field(self, client):
        """API v1 health response must include model_loaded boolean."""
        resp = client.get("/api/v1/health")
        assert "model_loaded" in resp.json()
        assert isinstance(resp.json()["model_loaded"], bool)

    def test_v1_health_has_window_sizes(self, client):
        """API v1 health response must include drift window size fields."""
        resp = client.get("/api/v1/health")
        body = resp.json()
        assert "reference_window_size" in body
        assert "current_window_size" in body


class TestVersionEndpoint:
    """Tests for the /version endpoint."""

    def test_version_returns_200(self, client):
        """Version endpoint returns HTTP 200."""
        resp = client.get("/version")
        assert resp.status_code == 200

    def test_version_has_version_key(self, client):
        """Version response contains a version key."""
        resp = client.get("/version")
        assert "version" in resp.json()


class TestPredictEndpoint:
    """Tests for POST /api/v1/predict."""

    def test_predict_returns_200_with_valid_payload(self, client, sample_metrics):
        """Predict endpoint accepts valid payload and returns 200."""
        resp = client.post("/api/v1/predict", json=sample_metrics)
        assert resp.status_code == 200

    def test_predict_response_has_required_fields(self, client, sample_metrics):
        """Predict response includes all required schema fields."""
        resp = client.post("/api/v1/predict", json=sample_metrics)
        body = resp.json()
        for field in ("service_name", "predicted_incident", "confidence", "model_version", "timestamp"):
            assert field in body, f"Missing field: {field}"

    def test_predict_confidence_in_range(self, client, sample_metrics):
        """Confidence score must be between 0 and 1."""
        resp = client.post("/api/v1/predict", json=sample_metrics)
        confidence = resp.json()["confidence"]
        assert 0.0 <= confidence <= 1.0

    def test_predict_service_name_echoed(self, client, sample_metrics):
        """Response service_name must match the request."""
        resp = client.post("/api/v1/predict", json=sample_metrics)
        assert resp.json()["service_name"] == sample_metrics["service_name"]

    @pytest.mark.parametrize("missing_field", [
        "cpu_usage_pct",
        "memory_usage_pct",
        "error_rate_per_min",
        "latency_p99_ms",
        "request_rate_per_sec",
        "disk_io_util_pct",
    ])
    def test_predict_rejects_missing_field(self, client, sample_metrics, missing_field):
        """Predict endpoint should return 422 when a required metric is missing."""
        payload = dict(sample_metrics)
        del payload[missing_field]
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 422

    @pytest.mark.parametrize("cpu_value,expected_status", [
        (0.0, 200),
        (50.0, 200),
        (100.0, 200),
        (-1.0, 422),
        (101.0, 422),
    ])
    def test_predict_cpu_boundary_validation(self, client, sample_metrics, cpu_value, expected_status):
        """CPU usage must be in [0, 100]."""
        payload = dict(sample_metrics)
        payload["cpu_usage_pct"] = cpu_value
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == expected_status

    def test_predict_empty_service_name_rejected(self, client, sample_metrics):
        """Empty service_name should return 422."""
        payload = dict(sample_metrics)
        payload["service_name"] = ""
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_normal_metrics_returns_low_confidence(self, client, normal_metrics):
        """Normal metrics should produce lower incident probability than stress metrics."""
        resp_normal = client.post("/api/v1/predict", json=normal_metrics)
        assert resp_normal.status_code == 200


class TestArtifactPairing:
    """Regression tests for model/feature-pipeline artifact pairing.

    The scaler inside the feature pipeline is stateful, so a persisted model is
    only usable alongside the pipeline it was trained with. An earlier version
    persisted the model but not the pipeline, so any warm start (model file
    present) paired the model with an unfitted scaler and every prediction
    failed with HTTP 422 — a fault that only appeared after a restart.
    """

    def test_load_artifacts_returns_fitted_pipeline(self, tmp_path, monkeypatch):
        """A cold bootstrap must persist BOTH artifacts, not just the model."""
        from pathlib import Path

        from app.config import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("MODEL_PATH", str(tmp_path / "m.pkl"))
        monkeypatch.setenv("FEATURE_PIPELINE_PATH", str(tmp_path / "p.pkl"))

        import app.api.v1.routes as routes

        monkeypatch.setattr(routes, "load_model", lambda: (_ for _ in ()).throw(FileNotFoundError()))

        model, pipeline = routes._load_artifacts()
        assert model is not None
        assert Path(tmp_path / "p.pkl").exists(), "feature pipeline was not persisted"
        get_settings.cache_clear()

    def test_predict_succeeds_on_warm_start(self, client, sample_metrics):
        """Predicting twice with a reloaded model must not raise NotFittedError."""
        import app.api.v1.routes as routes

        first = client.post("/api/v1/predict", json=sample_metrics)
        assert first.status_code == 200

        routes._model = None
        routes._feature_pipeline = None

        second = client.post("/api/v1/predict", json=sample_metrics)
        assert second.status_code == 200, (
            "warm start paired the model with an unfitted pipeline"
        )


class TestMetricsEndpoint:
    """Tests for GET /api/v1/metrics."""

    def test_metrics_returns_200(self, client):
        """Metrics endpoint returns HTTP 200."""
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200

    def test_metrics_has_required_fields(self, client):
        """Metrics response contains all required aggregate fields."""
        resp = client.get("/api/v1/metrics")
        body = resp.json()
        for field in ("total_predictions", "incident_count", "incident_rate", "drift_alerts_24h", "avg_confidence"):
            assert field in body

    def test_metrics_incident_rate_valid(self, client):
        """Incident rate must be between 0 and 1."""
        resp = client.get("/api/v1/metrics")
        rate = resp.json()["incident_rate"]
        assert 0.0 <= rate <= 1.0


class TestRunbookSearch:
    """Tests for POST /api/v1/runbooks/search."""

    def test_runbook_search_returns_200(self, client):
        """Runbook search returns HTTP 200."""
        resp = client.post(
            "/api/v1/runbooks/search",
            json={"query": "high cpu memory error", "top_k": 3},
        )
        assert resp.status_code == 200

    def test_runbook_search_returns_list(self, client):
        """Runbook search returns a list."""
        resp = client.post(
            "/api/v1/runbooks/search",
            json={"query": "network latency timeout", "top_k": 2},
        )
        assert isinstance(resp.json(), list)

    def test_runbook_search_query_too_short_rejected(self, client):
        """Query shorter than 3 characters should return 422."""
        resp = client.post(
            "/api/v1/runbooks/search",
            json={"query": "ab", "top_k": 3},
        )
        assert resp.status_code == 422


class TestForecastEndpoint:
    """Tests for GET /api/v1/forecast."""

    def test_forecast_returns_200(self, client):
        """Forecast endpoint returns HTTP 200."""
        resp = client.get("/api/v1/forecast")
        assert resp.status_code == 200

    def test_forecast_returns_24_points(self, client):
        """Forecast returns 24 hourly prediction points."""
        resp = client.get("/api/v1/forecast")
        assert len(resp.json()) == 24

    def test_forecast_points_have_required_fields(self, client):
        """Each forecast point has timestamp, value, lower_bound, upper_bound."""
        resp = client.get("/api/v1/forecast")
        for point in resp.json():
            for field in ("timestamp", "value", "lower_bound", "upper_bound"):
                assert field in point
