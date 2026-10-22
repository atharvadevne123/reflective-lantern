"""API endpoint tests for Social-Pulse."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_contains_model_version(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert "model_version" in resp.json()

    def test_health_contains_uptime(self, client: TestClient) -> None:
        resp = client.get("/health")
        data = resp.json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0.0


class TestVersionEndpoint:
    def test_version_returns_200(self, client: TestClient) -> None:
        resp = client.get("/version")
        assert resp.status_code == 200

    def test_version_contains_service(self, client: TestClient) -> None:
        resp = client.get("/version")
        assert resp.json()["service"] == "social-pulse"


class TestPredictEndpoint:
    def test_predict_valid_post(self, client: TestClient, sample_post: dict) -> None:
        resp = client.post("/predict", json=sample_post)
        assert resp.status_code in (200, 503)

    def test_predict_returns_probability(self, client: TestClient, sample_post: dict) -> None:
        resp = client.post("/predict", json=sample_post)
        if resp.status_code == 200:
            data = resp.json()
            assert 0.0 <= data["viral_probability"] <= 1.0

    def test_predict_returns_label(self, client: TestClient, sample_post: dict) -> None:
        resp = client.post("/predict", json=sample_post)
        if resp.status_code == 200:
            data = resp.json()
            assert isinstance(data["viral_label"], bool)

    def test_predict_returns_confidence(self, client: TestClient, sample_post: dict) -> None:
        resp = client.post("/predict", json=sample_post)
        if resp.status_code == 200:
            data = resp.json()
            assert 0.0 <= data["ensemble_confidence"] <= 1.0

    def test_predict_missing_text_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/predict", json={"author_followers": 100, "hour_of_day": 12, "day_of_week": 0}
        )
        assert resp.status_code == 422

    def test_predict_blank_text_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/predict",
            json={
                "text": "   ",
                "author_followers": 100,
                "hour_of_day": 12,
                "day_of_week": 0,
            },
        )
        assert resp.status_code == 422

    def test_predict_negative_followers_returns_422(
        self, client: TestClient, sample_post: dict
    ) -> None:
        bad = dict(sample_post)
        bad["author_followers"] = -1
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_invalid_hour_returns_422(self, client: TestClient, sample_post: dict) -> None:
        bad = dict(sample_post)
        bad["hour_of_day"] = 25
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_invalid_dow_returns_422(self, client: TestClient, sample_post: dict) -> None:
        bad = dict(sample_post)
        bad["day_of_week"] = 7
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_returns_request_id(self, client: TestClient, sample_post: dict) -> None:
        resp = client.post("/predict", json=sample_post)
        if resp.status_code == 200:
            assert "request_id" in resp.json()

    def test_predict_custom_request_id_echoed(self, client: TestClient, sample_post: dict) -> None:
        resp = client.post(
            "/predict",
            json=sample_post,
            headers={"X-Request-ID": "test-id-123"},
        )
        assert resp.headers.get("X-Request-ID") == "test-id-123"


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_contains_total_predictions(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        data = resp.json()
        assert "total_predictions" in data
        assert data["total_predictions"] >= 0

    def test_metrics_viral_rate_in_range(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        data = resp.json()
        assert 0.0 <= data["viral_rate"] <= 1.0


class TestDriftEndpoints:
    def test_drift_get_returns_200(self, client: TestClient) -> None:
        resp = client.get("/drift")
        assert resp.status_code == 200

    def test_drift_get_returns_list(self, client: TestClient) -> None:
        resp = client.get("/drift")
        data = resp.json()
        assert "drift_results" in data
        assert isinstance(data["drift_results"], list)

    def test_drift_check_post_returns_200(self, client: TestClient) -> None:
        resp = client.post("/drift/check")
        assert resp.status_code == 200


@pytest.mark.parametrize("hour", [0, 6, 12, 18, 23])
def test_predict_various_hours(client: TestClient, sample_post: dict, hour: int) -> None:
    """Predictions should succeed for any valid hour."""
    post = dict(sample_post)
    post["hour_of_day"] = hour
    resp = client.post("/predict", json=post)
    assert resp.status_code in (200, 503)


@pytest.mark.parametrize("followers", [0, 100, 10_000, 1_000_000, 50_000_000])
def test_predict_follower_buckets(client: TestClient, sample_post: dict, followers: int) -> None:
    """Predictions should succeed across all follower bucket boundaries."""
    post = dict(sample_post)
    post["author_followers"] = followers
    resp = client.post("/predict", json=post)
    assert resp.status_code in (200, 503)
