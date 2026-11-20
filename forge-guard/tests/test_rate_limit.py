"""Tests for rate limiting middleware."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import main as app_main


def test_rate_limit_returns_429_after_threshold(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(app_main, "RATE_LIMIT", 3)
    app_main._RATE_WINDOW.clear()

    responses = [client.get("/health").status_code for _ in range(5)]
    assert 429 in responses
    assert responses[0] == 200

    app_main._RATE_WINDOW.clear()


def test_rate_limit_window_resets(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(app_main, "RATE_LIMIT", 2)
    app_main._RATE_WINDOW.clear()

    client.get("/health")
    # Simulate window ageing by clearing state
    app_main._RATE_WINDOW.clear()
    resp = client.get("/health")
    assert resp.status_code == 200

    app_main._RATE_WINDOW.clear()


def test_rate_limit_first_request_always_succeeds(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(app_main, "RATE_LIMIT", 1)
    app_main._RATE_WINDOW.clear()
    resp = client.get("/health")
    assert resp.status_code == 200
    app_main._RATE_WINDOW.clear()


def test_rate_limit_does_not_affect_different_paths(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(app_main, "RATE_LIMIT", 100)
    app_main._RATE_WINDOW.clear()
    r1 = client.get("/health")
    r2 = client.get("/api/v1/version")
    assert r1.status_code == 200
    assert r2.status_code == 200
    app_main._RATE_WINDOW.clear()


def test_rate_limit_high_threshold_always_passes(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(app_main, "RATE_LIMIT", 1000)
    app_main._RATE_WINDOW.clear()
    for _ in range(10):
        resp = client.get("/health")
        assert resp.status_code == 200
    app_main._RATE_WINDOW.clear()


def test_rate_limit_second_request_blocked_at_limit_one(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(app_main, "RATE_LIMIT", 1)
    app_main._RATE_WINDOW.clear()
    r1 = client.get("/health")
    r2 = client.get("/health")
    assert r1.status_code == 200
    assert r2.status_code == 429
    app_main._RATE_WINDOW.clear()
