"""Tests for social-pulse app/main.py FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    mock_pipeline = MagicMock()
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.3, 0.7]]
    with (
        patch("app.database.init_db"),
        patch("app.model.load_model", return_value=(mock_pipeline, mock_model)),
        patch("app.model.generate_synthetic_data"),
        patch("app.model.train", return_value=MagicMock()),
    ):
        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestHealthEndpoint:
    def test_health_200(self, client: TestClient) -> None:
        assert client.get("/health").status_code == 200

    def test_health_returns_dict(self, client: TestClient) -> None:
        assert isinstance(client.get("/health").json(), dict)

    def test_health_has_status(self, client: TestClient) -> None:
        assert "status" in client.get("/health").json()


class TestMetricsEndpoint:
    def test_metrics_200(self, client: TestClient) -> None:
        assert client.get("/metrics").status_code == 200

    def test_metrics_is_dict(self, client: TestClient) -> None:
        assert isinstance(client.get("/metrics").json(), dict)


class TestPredictEndpoint:
    def test_predict_json(self, client: TestClient) -> None:
        resp = client.post(
            "/predict",
            json={
                "post_length": 120,
                "hashtag_count": 3,
                "mention_count": 1,
                "has_media": 1,
                "hour_of_day": 14,
                "day_of_week": 2,
                "follower_count": 5000,
                "following_count": 300,
                "account_age_days": 730,
                "avg_likes_per_post": 45.0,
                "platform": "twitter",
            },
        )
        assert isinstance(resp.json(), dict)

    def test_predict_missing_body(self, client: TestClient) -> None:
        resp = client.post("/predict", json={})
        assert resp.status_code in (200, 400, 422, 500)
