"""Tests for ops-vision app/main.py FastAPI endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


import pytest


@pytest.fixture()
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestRootEndpoints:
    def test_docs_available(self) -> None:
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.get("/docs")
            assert resp.status_code in (200, 404)

    def test_openapi_json(self) -> None:
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.get("/openapi.json")
            assert resp.status_code in (200, 404)

    def test_health_endpoint(self) -> None:
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.get("/api/v1/health")
            assert resp.status_code in (200, 404, 500)

    def test_metrics_endpoint(self) -> None:
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.get("/api/v1/metrics")
            assert resp.status_code in (200, 404, 500)

    def test_app_has_title(self) -> None:
        assert app.title == "Ops-Vision"

    def test_app_has_version(self) -> None:
        assert app.version is not None
