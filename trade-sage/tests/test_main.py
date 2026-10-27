"""Tests for trade-sage app/main.py FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.3, 0.7]]
    with (
        patch("app.database.init_db"),
        patch("app.model.load_model", return_value=mock_model),
        patch("app.model.read_champion_auc", return_value=0.85),
    ):
        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestHealthEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/health").status_code == 200

    def test_returns_dict(self, client: TestClient) -> None:
        assert isinstance(client.get("/health").json(), dict)

    def test_status_field(self, client: TestClient) -> None:
        assert "status" in client.get("/health").json()


class TestMetricsEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/metrics").status_code == 200


class TestPredictEndpoint:
    def test_returns_json(self, client: TestClient) -> None:
        resp = client.post(
            "/predict",
            json={
                "ticker": "AAPL",
                "open_price": 150.0,
                "high": 155.0,
                "low": 148.0,
                "close": 153.0,
                "volume": 5000000,
                "sma_20": 149.0,
                "sma_50": 145.0,
                "rsi_14": 60.0,
                "macd": 1.2,
                "macd_signal": 0.8,
                "bb_upper": 158.0,
                "bb_lower": 142.0,
                "obv": 1000000.0,
                "atr_14": 3.5,
                "pe_ratio": 28.0,
                "market_cap_log": 26.5,
                "sector_momentum": 0.03,
            },
        )
        assert isinstance(resp.json(), dict)
