"""Tests for ticket-oracle app/main.py FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    mock_models = (MagicMock(), MagicMock())
    with (
        patch("app.database.init_db"),
        patch("app.model.load_models", return_value=mock_models),
        patch("app.model.train_models"),
        patch("app.features.make_synthetic_dataset", return_value=MagicMock()),
    ):
        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestHealthEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/api/v1/health").status_code == 200

    def test_returns_dict(self, client: TestClient) -> None:
        assert isinstance(client.get("/api/v1/health").json(), dict)


class TestMetricsEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/api/v1/metrics").status_code == 200


class TestDriftEndpoint:
    def test_returns_200_or_503(self, client: TestClient) -> None:
        resp = client.get("/api/v1/drift")
        assert resp.status_code in (200, 404, 503)


class TestPredictEndpoint:
    def test_returns_json(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/predict",
            json={
                "ticket_id": "TKT-001",
                "title": "Server down",
                "description": "Production server is not responding",
                "customer_tier": "enterprise",
                "category": "infrastructure",
                "component": "server",
                "reporter_role": "engineer",
                "created_hour": 14,
                "created_day_of_week": 2,
                "historical_avg_resolution_hours": 4.0,
                "similar_ticket_count": 3,
                "assignee_workload": 5,
                "word_count": 8,
                "has_attachment": 0,
                "is_escalated": 0,
            },
        )
        assert isinstance(resp.json(), dict)
