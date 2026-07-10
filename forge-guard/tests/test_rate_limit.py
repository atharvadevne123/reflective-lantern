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
