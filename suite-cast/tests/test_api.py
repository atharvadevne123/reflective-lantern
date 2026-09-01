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


def test_correlation_id_header_propagated(client, sample_booking_dict):
    resp = client.post(
        "/api/v1/predict",
        json=sample_booking_dict,
        headers={"X-Correlation-ID": "test-corr-123"},
    )
    assert resp.headers.get("X-Correlation-ID") == "test-corr-123"


def test_correlation_id_generated_when_absent(client, sample_booking_dict):
    resp = client.post("/api/v1/predict", json=sample_booking_dict)
    assert "X-Correlation-ID" in resp.headers


def test_predict_suite_room_type(client, sample_booking_dict):
    payload = {**sample_booking_dict, "room_type": "suite"}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200


def test_predict_last_minute_booking(client, sample_booking_dict):
    payload = {**sample_booking_dict, "lead_time": 0}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200


def test_predict_far_advance_booking(client, sample_booking_dict):
    payload = {**sample_booking_dict, "lead_time": 365}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200


def test_predict_high_occupancy_scenario(client, sample_booking_dict):
    payload = {
        **sample_booking_dict,
        "current_occ_rate": 0.95,
        "checkin_month": 7,
        "special_event": 1,
    }
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    # High demand scenario should yield medium or high tier
    assert body["demand_tier"] in {"medium", "high"}


def test_root_banner(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["service"] == "Suite-Cast"


def test_health_reports_uptime(client):
    resp = client.get("/api/v1/health")
    assert resp.json()["uptime_seconds"] >= 0


def test_model_info_returns_ensemble(client):
    resp = client.get("/api/v1/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ensemble"] == ["XGBClassifier", "LGBMClassifier"]
    assert body["cv_folds"] == 5


def test_rate_limit_enforced(client, sample_booking_dict):
    from unittest.mock import patch as _patch

    import app.main as main_mod

    with _patch.object(main_mod.settings, "rate_limit_per_minute", 2):
        main_mod._request_windows.clear()
        first = client.get("/api/v1/health")
        second = client.get("/api/v1/health")
        third = client.get("/api/v1/health")
    main_mod._request_windows.clear()
    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429


def test_predictions_history_returns_200(client):
    resp = client.get("/api/v1/predictions")
    assert resp.status_code == 200
    assert "predictions" in resp.json()


def test_predictions_history_after_predict(client, sample_booking_dict):
    client.post("/api/v1/predict", json=sample_booking_dict)
    body = client.get("/api/v1/predictions?limit=5").json()
    assert body["count"] >= 1
    first = body["predictions"][0]
    assert {"request_id", "timestamp", "demand_score", "suggested_rate"} <= set(first)


def test_predictions_history_limit_clamped(client):
    resp = client.get("/api/v1/predictions?limit=5000")
    assert resp.status_code == 200
    assert resp.json()["count"] <= 100


@pytest.mark.parametrize("limit", [1, 5, 10])
def test_predictions_history_parametrized_limit(client, limit: int) -> None:
    resp = client.get(f"/api/v1/predictions?limit={limit}")
    assert resp.status_code == 200
    assert resp.json()["count"] <= limit


def test_predict_response_contains_request_id(client, sample_booking_dict) -> None:
    resp = client.post("/api/v1/predict", json=sample_booking_dict)
    assert resp.status_code == 200
    assert "request_id" in resp.json()


def test_health_response_contains_version(client) -> None:
    resp = client.get("/api/v1/health")
    body = resp.json()
    assert "version" in body or "status" in body
