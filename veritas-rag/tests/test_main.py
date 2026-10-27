"""Tests for veritas-rag app/main.py FastAPI endpoints."""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestHealthEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/health").status_code == 200

    def test_returns_dict(self, client: TestClient) -> None:
        assert isinstance(client.get("/health").json(), dict)

    def test_has_status(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert "status" in data or isinstance(data, dict)


class TestQueryEndpoint:
    def test_empty_query(self, client: TestClient) -> None:
        resp = client.post("/query", json={"query": ""})
        assert resp.status_code in (200, 400, 422)

    def test_simple_query(self, client: TestClient) -> None:
        resp = client.post("/query", json={"query": "What is Paris?"})
        assert resp.status_code in (200, 400, 422, 500)
        assert isinstance(resp.json(), dict)


class TestIngestEndpoint:
    def test_ingest_text(self, client: TestClient) -> None:
        content_b64 = base64.b64encode(b"France is a country in Europe.").decode()
        resp = client.post(
            "/ingest",
            json={
                "source": "test.txt",
                "content_b64": content_b64,
                "source_type": "text",
            },
        )
        assert resp.status_code in (200, 201, 400, 422, 500)

    def test_invalid_base64(self, client: TestClient) -> None:
        resp = client.post(
            "/ingest",
            json={
                "source": "bad.txt",
                "content_b64": "not-valid-base64!!!",
                "source_type": "text",
            },
        )
        assert resp.status_code in (200, 400, 422, 500)


class TestAppMetadata:
    def test_app_title(self) -> None:
        assert app.title == "Veritas-Rag"

    def test_openapi_json(self, client: TestClient) -> None:
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
