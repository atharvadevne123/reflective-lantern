"""API endpoint tests."""
from __future__ import annotations

import pytest

VALID_PAYLOAD = {
    "ticker": "AAPL",
    "close": 180.0,
    "open": 179.5,
    "high": 181.2,
    "low": 178.9,
    "volume": 5_000_000.0,
}


def test_health_ok(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_alpha_score(client):
    resp = client.post("/api/v1/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert "alpha_score" in body
    assert 0 <= body["alpha_score"] <= 100
    assert "prediction_id" in body
    assert body["ticker"] == "AAPL"


@pytest.mark.parametrize("ticker", ["MSFT", "GOOG", "TSLA"])
def test_predict_multiple_tickers(client, ticker):
    payload = {**VALID_PAYLOAD, "ticker": ticker, "close": 200.0, "open": 199.0, "high": 202.0, "low": 198.0}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["ticker"] == ticker


def test_predict_with_optional_fields(client):
    payload = {**VALID_PAYLOAD, "pe_ratio": 25.0, "market_cap": 2_900_000_000_000.0}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200


def test_predict_rejects_invalid_high_low(client):
    payload = {**VALID_PAYLOAD, "high": 100.0, "low": 200.0}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_zero_volume(client):
    payload = {**VALID_PAYLOAD, "volume": 0}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_metrics_endpoint(client):
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "total_predictions" in body


def test_drift_check_endpoint(client):
    import numpy as np
    rng = np.random.default_rng(1)
    values = rng.normal(50, 10, 50).tolist()
    resp = client.post("/api/v1/drift", json={"feature_name": "alpha_score", "values": values})
    assert resp.status_code == 200
    body = resp.json()
    assert "ks_statistic" in body
    assert "drift_detected" in body


def test_correlation_id_header(client):
    resp = client.get("/api/v1/health", headers={"X-Correlation-ID": "test-123"})
    assert resp.headers.get("X-Correlation-ID") == "test-123"


def test_rank_endpoint(client):
    securities = [
        {"ticker": "AAPL", "close": 180.0, "open": 179.5, "high": 181.2, "low": 178.9, "volume": 5_000_000.0},
        {"ticker": "MSFT", "close": 310.0, "open": 309.0, "high": 312.0, "low": 308.0, "volume": 3_000_000.0},
    ]
    resp = client.post("/api/v1/rank", json={"securities": securities})
    assert resp.status_code == 200
    ranked = resp.json()
    assert len(ranked) == 2
    assert ranked[0]["rank"] == 1
    assert ranked[0]["alpha_score"] >= ranked[1]["alpha_score"]


def test_rank_endpoint_single_security(client):
    securities = [
        {"ticker": "TSLA", "close": 220.0, "open": 219.0, "high": 222.0, "low": 218.0, "volume": 8_000_000.0},
    ]
    resp = client.post("/api/v1/rank", json={"securities": securities})
    assert resp.status_code == 200
    ranked = resp.json()
    assert ranked[0]["rank"] == 1
    assert ranked[0]["ticker"] == "TSLA"


def test_predict_missing_ticker(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "ticker"}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_predict_negative_close(client):
    payload = {**VALID_PAYLOAD, "close": -1.0}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_feature_importance_endpoint(client):
    resp = client.get("/api/v1/feature-importance")
    assert resp.status_code == 200
    body = resp.json()
    assert "feature_importances" in body
    assert len(body["feature_importances"]) > 0
    first = body["feature_importances"][0]
    assert "feature" in first and "importance" in first


def test_export_predictions_endpoint(client):
    client.post("/api/v1/predict", json=VALID_PAYLOAD)
    resp = client.get("/api/v1/predictions/export?limit=10")
    assert resp.status_code == 200
    body = resp.json()
    assert "count" in body
    assert "predictions" in body
    assert isinstance(body["predictions"], list)


def test_export_predictions_limit(client):
    resp = client.get("/api/v1/predictions/export?limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) <= 5
