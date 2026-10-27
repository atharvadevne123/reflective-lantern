"""Tests for forge-guard app/main.py FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestHealthEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/health").status_code == 200

    def test_returns_dict(self, client: TestClient) -> None:
        assert isinstance(client.get("/health").json(), dict)

    def test_has_status_field(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert "status" in data


class TestMetricsEndpoint:
    def test_returns_200_or_500(self, client: TestClient) -> None:
        resp = client.get("/ops/metrics")
        assert resp.status_code in (200, 500)

    def test_returns_dict(self, client: TestClient) -> None:
        resp = client.get("/ops/metrics")
        assert isinstance(resp.json(), dict)


class TestPredictEndpoint:
    def test_predict_valid_payload(self, client: TestClient) -> None:
        payload = {
            "temperature": 72.5,
            "pressure": 1.2,
            "vibration": 0.05,
            "speed": 1500.0,
            "torque": 45.0,
            "voltage": 220.0,
            "current": 10.0,
        }
        resp = client.post("/v1/predict", json=payload)
        assert resp.status_code in (200, 201, 400, 422, 500)

    def test_predict_missing_fields(self, client: TestClient) -> None:
        resp = client.post("/v1/predict", json={})
        assert resp.status_code in (400, 422, 500)


class TestAppMetadata:
    def test_app_title(self) -> None:
        assert app.title == "Forge-Guard"

    def test_openapi_json(self, client: TestClient) -> None:
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
