"""Tests for temporal-pulse app/main.py FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    with (
        patch("app.database.init_db"),
        patch("app.model.train_anomaly_detector"),
        patch("app.model.train_forecaster"),
        patch("app.model.get_model_version", return_value="1.0.0"),
        patch("app.model.score_anomaly", return_value=(False, 0.1)),
        patch("app.monitoring.log_prediction"),
    ):
        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestHealthEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/health").status_code == 200

    def test_returns_dict(self, client: TestClient) -> None:
        assert isinstance(client.get("/health").json(), dict)


class TestVersionEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/version")
        assert resp.status_code == 200

    def test_has_version_key(self, client: TestClient) -> None:
        data = client.get("/version").json()
        assert isinstance(data, dict)


class TestMetricsEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/metrics").status_code == 200


class TestAnomalyEndpoint:
    def test_returns_json(self, client: TestClient) -> None:
        resp = client.post(
            "/anomaly",
            json={
                "sensor_id": "s1",
                "timestamp": "2024-01-01T00:00:00",
                "value": 42.5,
                "window_values": [40.0, 41.0, 42.0, 43.0, 42.5],
            },
        )
        assert isinstance(resp.json(), dict)
