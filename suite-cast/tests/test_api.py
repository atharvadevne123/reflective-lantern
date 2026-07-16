"""Tests for the FastAPI endpoints in Suite-Cast."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(trained_models):
    """TestClient with model state injected, bypassing startup lifespan."""
    xgb_pipe, lgbm_pipe, metrics = trained_models

    with (
        patch("app.main._xgb_model", xgb_pipe),
        patch("app.main._lgbm_model", lgbm_pipe),
        patch("app.main._model_metrics", metrics),
        patch("app.main._reference_scores", [0.5] * 50),
        patch("app.database.engine"),
        patch("app.database.create_tables"),
    ):
        # Use in-memory SQLite for API tests
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool

        from app.database import Base

        test_engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(test_engine)
        TestSession = sessionmaker(bind=test_engine)

        from app.database import get_db
        from app.main import app

        def override_get_db():
            db = TestSession()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db

        yield TestClient(app, raise_server_exceptions=False)

        app.dependency_overrides.clear()


def test_health_returns_200(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200


def test_health_payload_keys(client):
    resp = client.get("/api/v1/health")
    body = resp.json()
    assert "status" in body
    assert "model_loaded" in body
    assert "model_version" in body


def test_health_status_healthy(client):
    resp = client.get("/api/v1/health")
    assert resp.json()["status"] == "healthy"


def test_predict_returns_200(client, sample_booking_dict):
    resp = client.post("/api/v1/predict", json=sample_booking_dict)
    assert resp.status_code == 200


def test_predict_response_has_demand_score(client, sample_booking_dict):
    resp = client.post("/api/v1/predict", json=sample_booking_dict)
    body = resp.json()
    assert "demand_score" in body
    assert 0.0 <= body["demand_score"] <= 1.0


def test_predict_response_has_suggested_rate(client, sample_booking_dict):
    resp = client.post("/api/v1/predict", json=sample_booking_dict)
    body = resp.json()
    assert "suggested_rate" in body
    assert body["suggested_rate"] > 0


def test_predict_response_has_request_id(client, sample_booking_dict):
    resp = client.post("/api/v1/predict", json=sample_booking_dict)
    body = resp.json()
    assert "request_id" in body
    assert len(body["request_id"]) == 36  # UUID format


def test_predict_demand_tier_valid(client, sample_booking_dict):
    resp = client.post("/api/v1/predict", json=sample_booking_dict)
    assert resp.json()["demand_tier"] in {"low", "medium", "high"}


def test_predict_invalid_lead_time(client, sample_booking_dict):
    payload = {**sample_booking_dict, "lead_time": -1}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_predict_invalid_room_type(client, sample_booking_dict):
    payload = {**sample_booking_dict, "room_type": "penthouse"}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_predict_invalid_channel(client, sample_booking_dict):
    payload = {**sample_booking_dict, "booking_channel": "carrier_pigeon"}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_predict_missing_required_field(client, sample_booking_dict):
    payload = {k: v for k, v in sample_booking_dict.items() if k != "lead_time"}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_metrics_endpoint_returns_200(client):
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200


def test_metrics_endpoint_has_drift_key(client):
    body = client.get("/api/v1/metrics").json()
    assert "drift" in body
    assert "ks_statistic" in body["drift"]
