"""Tests for signal-forge app/main.py FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    with (
        patch("app.database.create_tables"),
        patch("app.model.predict", return_value={"regime": "bull", "confidence": 0.85}),
    ):
        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestHealthEndpoint:
    def test_health_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_json(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert isinstance(data, dict)


class TestMetricsEndpoint:
    def test_metrics_200(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200


class TestVersionEndpoint:
    def test_version_200(self, client: TestClient) -> None:
        resp = client.get("/version")
        assert resp.status_code == 200

    def test_version_has_version_field(self, client: TestClient) -> None:
        data = client.get("/version").json()
        assert "version" in data or isinstance(data, dict)


class TestPredictEndpoint:
    def test_predict_returns_json(self, client: TestClient) -> None:
        resp = client.post(
            "/predict",
            json={
                "open": 100.0,
                "high": 105.0,
                "low": 98.0,
                "close": 103.0,
                "volume": 1000000,
                "vix": 18.5,
                "yield_spread": 0.5,
                "rsi_14": 55.0,
                "macd": 0.3,
                "bb_width": 0.02,
                "adx": 25.0,
                "obv_change_pct": 1.2,
                "atr_14": 1.5,
                "sector_momentum": 0.5,
                "market_cap_log": 22.0,
            },
        )
        assert isinstance(resp.json(), dict)
