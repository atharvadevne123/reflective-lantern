"""Tests for sport-cast app/main.py FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    mock_model = MagicMock()
    with (
        patch("app.database.init_db"),
        patch("app.model.load_model", return_value=(mock_model, MagicMock())),
    ):
        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestHealthEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_returns_dict(self, client: TestClient) -> None:
        assert isinstance(client.get("/health").json(), dict)


class TestMetricsEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200


class TestPredictEndpoint:
    def test_returns_json(self, client: TestClient) -> None:
        resp = client.post(
            "/predict",
            json={
                "home_team": "TeamA",
                "away_team": "TeamB",
                "home_goals_avg": 1.8,
                "away_goals_avg": 1.2,
                "home_win_rate": 0.6,
                "away_win_rate": 0.4,
                "head_to_head_home_wins": 3,
                "head_to_head_draws": 2,
                "head_to_head_away_wins": 1,
                "home_form_points": 10,
                "away_form_points": 7,
                "home_injuries": 1,
                "away_injuries": 2,
                "neutral_venue": 0,
            },
        )
        assert isinstance(resp.json(), dict)
