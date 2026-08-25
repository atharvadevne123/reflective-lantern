"""API endpoint tests for Quake-Net FastAPI application."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.conftest import VALID_EVENT


class TestHealthEndpoint:
    def test_health_returns_ok(self, app_client: TestClient) -> None:
        resp = app_client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_health_has_version(self, app_client: TestClient) -> None:
        resp = app_client.get("/api/v1/health")
        assert resp.json()["version"] == "1.0.0"

    def test_root_redirects(self, app_client: TestClient) -> None:
        resp = app_client.get("/")
        assert resp.status_code == 200
        assert "docs" in resp.json()


class TestPredictEndpoint:
    def test_predict_valid_event(self, app_client: TestClient) -> None:
        resp = app_client.post("/api/v1/predict", json=VALID_EVENT)
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_magnitude" in data
        assert 0.1 <= data["predicted_magnitude"] <= 9.9

    def test_predict_returns_aftershock_probability(self, app_client: TestClient) -> None:
        resp = app_client.post("/api/v1/predict", json=VALID_EVENT)
        assert resp.status_code == 200
        prob = resp.json()["aftershock_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_returns_magnitude_class(self, app_client: TestClient) -> None:
        resp = app_client.post("/api/v1/predict", json=VALID_EVENT)
        assert resp.json()["magnitude_class"] in [
            "micro",
            "minor",
            "light",
            "moderate",
            "strong",
            "major",
            "great",
        ]

    def test_predict_sets_correlation_id_header(self, app_client: TestClient) -> None:
        resp = app_client.post("/api/v1/predict", json=VALID_EVENT)
        assert "x-correlation-id" in resp.headers

    @pytest.mark.parametrize(
        "fault_type", ["strike_slip", "reverse", "normal", "oblique", "unknown"]
    )
    def test_predict_all_fault_types(self, app_client: TestClient, fault_type: str) -> None:
        event = {**VALID_EVENT, "fault_type": fault_type}
        resp = app_client.post("/api/v1/predict", json=event)
        assert resp.status_code == 200

    def test_predict_rejects_invalid_latitude(self, app_client: TestClient) -> None:
        event = {**VALID_EVENT, "latitude": 999.0}
        resp = app_client.post("/api/v1/predict", json=event)
        assert resp.status_code == 422

    def test_predict_rejects_invalid_fault_type(self, app_client: TestClient) -> None:
        event = {**VALID_EVENT, "fault_type": "dragon_fault"}
        resp = app_client.post("/api/v1/predict", json=event)
        assert resp.status_code == 422

    def test_predict_rejects_zero_depth(self, app_client: TestClient) -> None:
        event = {**VALID_EVENT, "depth_km": 0.0}
        resp = app_client.post("/api/v1/predict", json=event)
        assert resp.status_code == 422

    def test_predict_rejects_negative_amplitude(self, app_client: TestClient) -> None:
        event = {**VALID_EVENT, "p_wave_amplitude": -1.0}
        resp = app_client.post("/api/v1/predict", json=event)
        assert resp.status_code == 422

    def test_predict_increments_counter(self, app_client: TestClient) -> None:
        from app.main import _counters

        before = _counters["predictions"]
        app_client.post("/api/v1/predict", json=VALID_EVENT)
        assert _counters["predictions"] == before + 1


class TestBatchPredictEndpoint:
    def test_batch_predict_single(self, app_client: TestClient) -> None:
        resp = app_client.post("/api/v1/predict/batch", json={"events": [VALID_EVENT]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert len(data["results"]) == 1

    def test_batch_predict_multiple(self, app_client: TestClient) -> None:
        events = [VALID_EVENT] * 5
        resp = app_client.post("/api/v1/predict/batch", json={"events": events})
        assert resp.status_code == 200
        assert resp.json()["count"] == 5

    def test_batch_predict_empty_rejected(self, app_client: TestClient) -> None:
        resp = app_client.post("/api/v1/predict/batch", json={"events": []})
        assert resp.status_code == 422


class TestMetricsEndpoint:
    def test_metrics_returns_counters(self, app_client: TestClient) -> None:
        resp = app_client.get("/api/v1/metrics")
        assert resp.status_code == 200
        assert "service_counters" in resp.json()

    def test_metrics_contains_rate_limit(self, app_client: TestClient) -> None:
        resp = app_client.get("/api/v1/metrics")
        assert "rate_limit_per_min" in resp.json()


class TestDriftEndpoint:
    def test_drift_report_structure(self, app_client: TestClient) -> None:
        resp = app_client.get("/api/v1/drift")
        assert resp.status_code == 200
        data = resp.json()
        assert "feature_drifts" in data
        assert "drift_detected_count" in data

    def test_drift_psi_endpoint(self, app_client: TestClient) -> None:
        resp = app_client.get("/api/v1/drift/psi")
        assert resp.status_code == 200
        assert "psi_scores" in resp.json()


class TestRecentEventsEndpoint:
    def test_recent_events_after_prediction(self, app_client: TestClient) -> None:
        app_client.post("/api/v1/predict", json=VALID_EVENT)
        resp = app_client.get("/api/v1/events/recent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1

    def test_recent_events_limit_capped(self, app_client: TestClient) -> None:
        resp = app_client.get("/api/v1/events/recent?limit=5")
        assert resp.status_code == 200
        assert resp.json()["count"] <= 5


class TestSimilarityEndpoint:
    def test_similar_without_history_returns_note(self, app_client: TestClient) -> None:
        resp = app_client.post("/api/v1/similar", json=VALID_EVENT)
        assert resp.status_code == 200
        assert "matches" in resp.json()

    def test_similar_after_predictions(self, app_client: TestClient) -> None:
        for _ in range(3):
            app_client.post("/api/v1/predict", json=VALID_EVENT)
        resp = app_client.post("/api/v1/similar?limit=2", json=VALID_EVENT)
        assert resp.status_code == 200
        assert resp.json()["count"] <= 2

    def test_similar_rejects_invalid_payload(self, app_client: TestClient) -> None:
        resp = app_client.post("/api/v1/similar", json={"latitude": 1.0})
        assert resp.status_code == 422


class TestAnomaliesEndpoint:
    def test_anomalies_endpoint_responds(self, app_client: TestClient) -> None:
        resp = app_client.get("/api/v1/anomalies")
        assert resp.status_code == 200
        assert "anomalies" in resp.json()

    def test_anomalies_reports_scored_count(self, app_client: TestClient) -> None:
        resp = app_client.get("/api/v1/anomalies?limit=50")
        assert resp.status_code == 200
        body = resp.json()
        assert "count" in body


class TestCacheStatsEndpoint:
    def test_cache_stats_shape(self, app_client: TestClient) -> None:
        resp = app_client.get("/api/v1/cache/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert "hit_rate" in body
        assert "entries" in body


class TestOpenAPISchema:
    def test_openapi_available(self, app_client: TestClient) -> None:
        resp = app_client.get("/openapi.json")
        assert resp.status_code == 200

    def test_all_routes_versioned(self, app_client: TestClient) -> None:
        paths = app_client.get("/openapi.json").json()["paths"]
        non_versioned = [p for p in paths if not p.startswith("/api/v1")]
        assert non_versioned == [], f"Unversioned routes: {non_versioned}"

    def test_endpoints_have_summaries_or_descriptions(self, app_client: TestClient) -> None:
        paths = app_client.get("/openapi.json").json()["paths"]
        undocumented = [
            f"{method} {path}"
            for path, methods in paths.items()
            for method, spec in methods.items()
            if not (spec.get("summary") or spec.get("description"))
        ]
        assert undocumented == [], f"Undocumented: {undocumented}"


class TestForecastEndpoint:
    def test_forecast_returns_daily_entries(self, app_client: TestClient) -> None:
        resp = app_client.post(
            "/api/v1/forecast/aftershocks",
            json={"mainshock_magnitude": 6.5, "horizon_days": 7},
        )
        assert resp.status_code == 200
        assert len(resp.json()["daily_forecast"]) == 7

    def test_forecast_includes_half_life(self, app_client: TestClient) -> None:
        resp = app_client.post("/api/v1/forecast/aftershocks", json={"mainshock_magnitude": 7.0})
        assert resp.status_code == 200
        assert "decay_half_life_days" in resp.json()

    def test_forecast_fits_observed_sequence(self, app_client: TestClient) -> None:
        resp = app_client.post(
            "/api/v1/forecast/aftershocks",
            json={
                "mainshock_magnitude": 6.0,
                "horizon_days": 5,
                "observed_times_days": [0.1, 0.4, 1.0, 2.2, 4.5],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["omori_parameters"]["fitted"] is True

    def test_forecast_rejects_zero_horizon(self, app_client: TestClient) -> None:
        resp = app_client.post(
            "/api/v1/forecast/aftershocks",
            json={"mainshock_magnitude": 6.0, "horizon_days": 0},
        )
        assert resp.status_code == 422

    def test_forecast_rejects_out_of_range_magnitude(self, app_client: TestClient) -> None:
        resp = app_client.post("/api/v1/forecast/aftershocks", json={"mainshock_magnitude": 42.0})
        assert resp.status_code == 422


class TestHealthDiagnostics:
    def test_health_reports_database_reachable(self, app_client: TestClient) -> None:
        resp = app_client.get("/api/v1/health")
        assert resp.json()["database_reachable"] is True

    def test_health_reports_uptime(self, app_client: TestClient) -> None:
        resp = app_client.get("/api/v1/health")
        assert resp.json()["uptime_seconds"] >= 0.0

    def test_health_degrades_without_model(self, app_client: TestClient) -> None:
        from app.main import _model_cache

        pipeline = _model_cache.pop("pipeline", None)
        try:
            body = app_client.get("/api/v1/health").json()
            assert body["status"] == "degraded"
            assert body["model_loaded"] is False
        finally:
            if pipeline is not None:
                _model_cache["pipeline"] = pipeline


class TestBatchFailureReporting:
    def test_successful_batch_has_no_failures(self, app_client: TestClient) -> None:
        resp = app_client.post("/api/v1/predict/batch", json={"events": [VALID_EVENT] * 3})
        body = resp.json()
        assert body["errors"] == 0
        assert body["failures"] == []

    def test_failure_carries_index(self, app_client: TestClient, monkeypatch) -> None:
        import app.main as main_module

        calls = {"n": 0}
        real = main_module.predict_magnitude

        def flaky(pipeline, features):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("synthetic failure")
            return real(pipeline, features)

        monkeypatch.setattr(main_module, "predict_magnitude", flaky)
        resp = app_client.post("/api/v1/predict/batch", json={"events": [VALID_EVENT] * 3})
        body = resp.json()
        assert body["errors"] == 1
        assert body["failures"][0]["index"] == 1
        assert body["count"] == 2
