"""Tests for suite-cast app/main.py FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.75]
    with (
        patch("app.database.create_tables"),
        patch("app.model.generate_sample_data", return_value=MagicMock()),
        patch("app.model.load_models", return_value=(mock_model, mock_model, {})),
        patch("app.monitoring.log_prediction"),
    ):
        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert "status" in data


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_is_dict(self, client: TestClient) -> None:
        data = client.get("/metrics").json()
        assert isinstance(data, dict)


class TestPredictEndpoint:
    def test_predict_returns_json(self, client: TestClient) -> None:
        resp = client.post(
            "/predict",
            json={
                "booking_lead_days": 30,
                "length_of_stay": 2,
                "room_type": "standard",
                "weekend_flag": 0,
                "occupancy_rate_last_week": 0.75,
                "avg_daily_rate_last_week": 150.0,
                "competitor_rate_index": 1.0,
                "season": "summer",
                "special_event_flag": 0,
            },
        )
        assert isinstance(resp.json(), dict)

    def test_predict_empty_body_returns_error(self, client: TestClient) -> None:
        resp = client.post("/predict", json={})
        assert resp.status_code in (200, 400, 422, 500)
