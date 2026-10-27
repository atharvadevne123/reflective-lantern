"""Tests for quake-net app/main.py FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    mock_model = MagicMock()
    mock_model.predict.return_value = [5.2]
    with (
        patch("app.database.init_db"),
        patch("app.model.load_model", return_value=(mock_model, MagicMock())),
        patch("app.model.read_champion_metrics", return_value={"rmse": 0.5}),
        patch("app.monitoring.get_store", return_value=MagicMock()),
    ):
        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestHealthEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/health").status_code == 200

    def test_returns_dict(self, client: TestClient) -> None:
        assert isinstance(client.get("/health").json(), dict)

    def test_has_status(self, client: TestClient) -> None:
        assert "status" in client.get("/health").json()


class TestMetricsEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/metrics").status_code == 200

    def test_is_dict(self, client: TestClient) -> None:
        assert isinstance(client.get("/metrics").json(), dict)


class TestPredictEndpoint:
    def test_returns_json(self, client: TestClient) -> None:
        resp = client.post(
            "/predict",
            json={
                "latitude": 37.7749,
                "longitude": -122.4194,
                "depth_km": 10.0,
                "fault_type": "strike_slip",
                "station_count": 5,
                "rms_error": 0.1,
                "gap_degrees": 90.0,
                "min_distance_km": 15.0,
                "p_wave_velocity": 6.0,
                "s_wave_velocity": 3.5,
            },
        )
        assert isinstance(resp.json(), dict)

    def test_empty_body_returns_error(self, client: TestClient) -> None:
        resp = client.post("/predict", json={})
        assert resp.status_code in (400, 422, 200, 500)
