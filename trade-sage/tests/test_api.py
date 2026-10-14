"""Tests for FastAPI endpoints."""
from __future__ import annotations

import os

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def set_tmp_model_paths(tmp_path):
    os.environ["MODEL_PATH"] = str(tmp_path / "model.joblib")
    os.environ["METRICS_PATH"] = str(tmp_path / "metrics.json")
    yield
    os.environ.pop("MODEL_PATH", None)
    os.environ.pop("METRICS_PATH", None)


@pytest.fixture()
def client(db_session):
    from fastapi.testclient import TestClient

    from app.database import get_db
    from app.main import app

    def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _make_ohlcv(n=100):
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return [
        {
            "open": float(close[i] + rng.normal(0, 0.3)),
            "high": float(close[i] + rng.uniform(0.5, 2)),
            "low": float(close[i] - rng.uniform(0.5, 2)),
            "close": float(close[i]),
            "volume": float(rng.integers(1_000_000, 5_000_000)),
        }
        for i in range(n)
    ]


def test_health_returns_200(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("healthy", "degraded")


def test_health_has_required_fields(client):
    resp = client.get("/api/v1/health")
    body = resp.json()
    assert "model_loaded" in body
    assert "db_connected" in body
    assert "version" in body


def test_metrics_endpoint(client):
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "total_predictions" in body
    assert "uptime_seconds" in body


def test_train_endpoint(client):
    resp = client.post("/api/v1/train")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "trained"
    assert "auc_mean" in body["metrics"]


def test_predict_returns_signal(client):
    client.post("/api/v1/train")
    ohlcv = _make_ohlcv(100)
    resp = client.post("/api/v1/predict", json={"ticker": "AAPL", "horizon_days": 5, "ohlcv": ohlcv})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "AAPL"
    assert body["signal"] in ("BUY", "SELL")
    assert 0 <= body["confidence"] <= 1


def test_predict_without_model_503(client):
    # Force model to None
    import app.main as m
    prev = m._model
    m._model = None
    ohlcv = _make_ohlcv(100)
    resp = client.post("/api/v1/predict", json={"ticker": "TSLA", "horizon_days": 5, "ohlcv": ohlcv})
    assert resp.status_code == 503
    m._model = prev


def test_predict_insufficient_rows_422(client):
    client.post("/api/v1/train")
    ohlcv = _make_ohlcv(30)  # < 60 minimum
    resp = client.post("/api/v1/predict", json={"ticker": "AAPL", "horizon_days": 5, "ohlcv": ohlcv})
    assert resp.status_code == 422


def test_portfolio_risk_endpoint(client):
    ohlcv = _make_ohlcv(100)
    payload = {
        "holdings": [
            {"ticker": "AAPL", "weight": 0.6, "ohlcv": ohlcv},
            {"ticker": "GOOG", "weight": 0.4, "ohlcv": ohlcv},
        ],
        "confidence_level": 0.95,
    }
    resp = client.post("/api/v1/portfolio/risk", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "value_at_risk" in body
    assert "portfolio_volatility" in body


def test_portfolio_weights_must_sum_to_one(client):
    ohlcv = _make_ohlcv(100)
    payload = {
        "holdings": [
            {"ticker": "AAPL", "weight": 0.5, "ohlcv": ohlcv},
            {"ticker": "GOOG", "weight": 0.3, "ohlcv": ohlcv},
        ],
    }
    resp = client.post("/api/v1/portfolio/risk", json=payload)
    assert resp.status_code == 422


def test_drift_endpoint(client):
    ohlcv = _make_ohlcv(100)
    resp = client.post("/api/v1/drift", json={"ohlcv": ohlcv})
    assert resp.status_code == 200
    body = resp.json()
    assert "drift_results" in body
    assert "features_drifted" in body


def test_correlation_id_header(client):
    resp = client.get("/api/v1/health", headers={"X-Correlation-ID": "test-id-123"})
    assert resp.headers.get("x-correlation-id") == "test-id-123"


@pytest.mark.parametrize("horizon", [1, 5, 30, 60])
def test_predict_various_horizons(client, horizon):
    client.post("/api/v1/train")
    ohlcv = _make_ohlcv(100)
    resp = client.post(
        "/api/v1/predict",
        json={"ticker": "MSFT", "horizon_days": horizon, "ohlcv": ohlcv},
    )
    assert resp.status_code == 200
    assert resp.json()["horizon_days"] == horizon
