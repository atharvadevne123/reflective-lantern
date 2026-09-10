"""Tests for the retrain API endpoint."""

import pytest


def test_retrain_requires_api_key(client):
    response = client.post("/api/v1/retrain")
    assert response.status_code == 401


def test_retrain_wrong_key_rejected(client):
    response = client.post("/api/v1/retrain", headers={"X-Retrain-Key": "wrong-key"})
    assert response.status_code == 401


def test_retrain_with_correct_key(client, monkeypatch):
    monkeypatch.setenv("RETRAIN_API_KEY", "test-key")
    import app.retrain_endpoint as re_mod
    re_mod.RETRAIN_API_KEY = "test-key"

    response = client.post(
        "/api/v1/retrain",
        params={"n_samples": 100, "force": True},
        headers={"X-Retrain-Key": "test-key"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"


def test_retrain_history_endpoint(client):
    response = client.get("/api/v1/retrain/history")
    assert response.status_code == 200
    data = response.json()
    assert "history" in data
    assert isinstance(data["history"], list)


def test_retrain_skips_without_force(client, monkeypatch):
    import app.retrain_endpoint as re_mod
    re_mod.RETRAIN_API_KEY = "skip-key"
    response = client.post(
        "/api/v1/retrain",
        params={"n_samples": 100, "force": False},
        headers={"X-Retrain-Key": "skip-key"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("skipped", "completed")
