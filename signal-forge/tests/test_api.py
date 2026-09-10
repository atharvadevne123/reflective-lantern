"""API endpoint tests for Signal-Forge."""

import pytest


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_version(client):
    resp = client.get("/version")
    assert resp.status_code == 200
    assert "version" in resp.json()


def test_metrics_returns_uptime(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "uptime_seconds" in data
    assert "request_count" in data


def test_predict_valid(client, predict_payload):
    resp = client.post("/predict", json=predict_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "AAPL"
    assert data["regime"] in ("bull", "bear", "sideways", "volatile")
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["risk_score"] <= 1.0


def test_predict_lowercase_ticker(client):
    resp = client.post("/predict", json={"ticker": "msft", "close": 400.0, "volume": 20_000_000.0})
    assert resp.status_code == 200
    assert resp.json()["ticker"] == "MSFT"


def test_predict_with_market_return(client):
    payload = {"ticker": "SPY", "close": 500.0, "volume": 100_000_000.0, "market_return": 0.005}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["regime"] in ("bull", "bear", "sideways", "volatile")


def test_predict_missing_ticker(client):
    resp = client.post("/predict", json={"close": 100.0, "volume": 1_000_000.0})
    assert resp.status_code == 422


def test_predict_zero_close(client):
    resp = client.post("/predict", json={"ticker": "BAD", "close": 0.0, "volume": 1_000_000.0})
    assert resp.status_code == 422


def test_predict_negative_close(client):
    resp = client.post("/predict", json={"ticker": "BAD", "close": -10.0, "volume": 1_000_000.0})
    assert resp.status_code == 422


def test_predict_empty_ticker(client):
    resp = client.post("/predict", json={"ticker": "", "close": 100.0, "volume": 1_000_000.0})
    assert resp.status_code == 422


def test_drift_check(client):
    resp = client.post("/drift-check")
    assert resp.status_code == 200
    data = resp.json()
    assert "drift_detected" in data
    assert "features" in data


@pytest.mark.parametrize("ticker,close,volume", [
    ("GOOGL", 150.0, 30_000_000.0),
    ("TSLA", 250.0, 50_000_000.0),
    ("AMZN", 180.0, 40_000_000.0),
])
def test_predict_multiple_tickers(client, ticker, close, volume):
    resp = client.post("/predict", json={"ticker": ticker, "close": close, "volume": volume})
    assert resp.status_code == 200
    assert resp.json()["ticker"] == ticker
