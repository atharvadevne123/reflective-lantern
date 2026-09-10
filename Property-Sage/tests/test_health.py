"""Tests for the health check endpoints."""

import pytest

from app.health import deep_health_check


def test_liveness_health(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"


def test_deep_health_endpoint(client):
    response = client.get("/api/v1/health/deep")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "components" in data
    assert "database" in data["components"]


def test_deep_health_db_ok(db_session):
    result = deep_health_check(
        db=db_session,
        price_model_loaded=True,
        rental_model_loaded=True,
    )
    assert result["components"]["database"] == "ok"
    assert result["components"]["price_model_loaded"] == "ok"
    assert result["components"]["rental_model_loaded"] == "ok"


def test_deep_health_degraded_when_model_not_loaded(db_session):
    result = deep_health_check(
        db=db_session,
        price_model_loaded=False,
        rental_model_loaded=True,
    )
    assert result["status"] == "degraded"
    assert result["components"]["price_model_loaded"] == "not_loaded"
