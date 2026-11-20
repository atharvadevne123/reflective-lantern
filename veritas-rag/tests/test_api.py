"""HTTP API tests."""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, empty_pipeline):
    import app.main as main_mod

    monkeypatch.setattr(main_mod, "pipeline", empty_pipeline)
    return TestClient(main_mod.app)


def b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_root_banner(client):
    assert client.get("/").json()["service"] == "Veritas-Rag"


def test_ingest_then_query_with_citations(client):
    resp = client.post(
        "/api/v1/ingest",
        json={
            "source": "facts.txt",
            "content_base64": b64("The service level agreement guarantees 99.9 percent uptime."),
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ingested"

    resp = client.post(
        "/api/v1/query",
        json={"question": "What uptime does the service level agreement guarantee?"},
    )
    body = resp.json()
    assert body["answered"] is True
    assert "99.9" in body["answer"]
    assert body["citations"]
    assert body["citations"][0]["source"] == "facts.txt"


def test_query_insufficient_evidence(client):
    resp = client.post("/api/v1/query", json={"question": "What is the airspeed of a swallow?"})
    body = resp.json()
    assert body["answered"] is False
    assert "Insufficient evidence" in body["answer"]


def test_invalid_base64_rejected(client):
    resp = client.post(
        "/api/v1/ingest", json={"source": "x.txt", "content_base64": "!!!not-base64!!!"}
    )
    assert resp.status_code == 422


def test_short_question_rejected(client):
    resp = client.post("/api/v1/query", json={"question": "hi"})
    assert resp.status_code == 422


def test_invalid_source_type_rejected(client):
    resp = client.post(
        "/api/v1/ingest",
        json={"source": "x.bin", "content_base64": b64("data"), "source_type": "binary"},
    )
    assert resp.status_code == 422


def test_metrics_endpoint(client):
    client.post(
        "/api/v1/ingest", json={"source": "m.txt", "content_base64": b64("metric content here")}
    )
    body = client.get("/api/v1/metrics").json()
    assert body["documents"] == 1
    assert "cache" in body


def test_trace_endpoint_roundtrip(client):
    client.post(
        "/api/v1/ingest",
        json={"source": "t.txt", "content_base64": b64("Tracing target sentence lives here.")},
    )
    query = client.post(
        "/api/v1/query", json={"question": "Where does the tracing target sentence live?"}
    ).json()
    trace = client.get(f"/api/v1/trace/{query['request_id']}")
    assert trace.status_code == 200
    assert trace.json()["query"]


def test_trace_not_found(client):
    assert client.get("/api/v1/trace/nonexistent").status_code == 404


def test_traces_listing(client):
    client.post("/api/v1/query", json={"question": "list me in traces"})
    body = client.get("/api/v1/traces?limit=5").json()
    assert len(body["traces"]) >= 1
