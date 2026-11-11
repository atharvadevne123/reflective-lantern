"""API endpoint tests."""
from __future__ import annotations

import pytest

VALID_PAYLOAD = {
    "ticker": "AAPL", "close": 180.0, "open": 179.5, "high": 181.2, "low": 178.9, "volume": 5_000_000.0,
}


def test_health_ok(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_returns_alpha_score(client):
    resp = client.post("/api/v1/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["alpha_score"] <= 100
    assert body["ticker"] == "AAPL"


@pytest.mark.parametrize("ticker", ["MSFT", "GOOG", "TSLA"])
def test_predict_multiple_tickers(client, ticker):
    payload = {**VALID_PAYLOAD, "ticker": ticker, "close": 200.0, "open": 199.0, "high": 202.0, "low": 198.0}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200


def test_predict_rejects_invalid_high_low(client):
    payload = {**VALID_PAYLOAD, "high": 100.0, "low": 200.0}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_zero_volume(client):
    assert client.post("/api/v1/predict", json={**VALID_PAYLOAD, "volume": 0}).status_code == 422


def test_metrics_endpoint(client):
    assert "total_predictions" in client.get("/api/v1/metrics").json()


def test_rank_endpoint(client):
    secs = [
        {"ticker": "AAPL", "close": 180.0, "open": 179.5, "high": 181.2, "low": 178.9, "volume": 5_000_000.0},
        {"ticker": "MSFT", "close": 310.0, "open": 309.0, "high": 312.0, "low": 308.0, "volume": 3_000_000.0},
    ]
    resp = client.post("/api/v1/rank", json={"securities": secs})
    assert resp.status_code == 200
    ranked = resp.json()
    assert ranked[0]["rank"] == 1
    assert ranked[0]["alpha_score"] >= ranked[1]["alpha_score"]


def test_feature_importance_endpoint(client):
    resp = client.get("/api/v1/feature-importance")
    assert resp.status_code == 200
    assert "feature_importances" in resp.json()


def test_export_predictions_endpoint(client):
    client.post("/api/v1/predict", json=VALID_PAYLOAD)
    resp = client.get("/api/v1/predictions/export?limit=10")
    assert resp.status_code == 200
    assert "predictions" in resp.json()
