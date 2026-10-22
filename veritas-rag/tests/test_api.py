"""HTTP API tests."""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, empty_pipeline) -> None:
    import app.main as main_mod

    monkeypatch.setattr(main_mod, "pipeline", empty_pipeline)
    return TestClient(main_mod.app)


def b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def test_health(client) -> None:
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_root_banner(client) -> None:
    assert client.get("/").json()["service"] == "Veritas-Rag"


def test_ingest_then_query_with_citations(client) -> None:
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


def test_query_insufficient_evidence(client) -> None:
    resp = client.post("/api/v1/query", json={"question": "What is the airspeed of a swallow?"})
    body = resp.json()
    assert body["answered"] is False
    assert "Insufficient evidence" in body["answer"]


def test_invalid_base64_rejected(client) -> None:
    resp = client.post(
        "/api/v1/ingest", json={"source": "x.txt", "content_base64": "!!!not-base64!!!"}
    )
    assert resp.status_code == 422


def test_short_question_rejected(client) -> None:
    resp = client.post("/api/v1/query", json={"question": "hi"})
    assert resp.status_code == 422


def test_invalid_source_type_rejected(client) -> None:
    resp = client.post(
        "/api/v1/ingest",
        json={"source": "x.bin", "content_base64": b64("data"), "source_type": "binary"},
    )
    assert resp.status_code == 422


def test_metrics_endpoint(client) -> None:
    client.post(
        "/api/v1/ingest", json={"source": "m.txt", "content_base64": b64("metric content here")}
    )
    body = client.get("/api/v1/metrics").json()
    assert body["documents"] == 1
    assert "cache" in body


def test_trace_endpoint_roundtrip(client) -> None:
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


def test_trace_not_found(client) -> None:
    assert client.get("/api/v1/trace/nonexistent").status_code == 404


def test_traces_listing(client) -> None:
    client.post("/api/v1/query", json={"question": "list me in traces"})
    body = client.get("/api/v1/traces?limit=5").json()
    assert len(body["traces"]) >= 1


@pytest.mark.parametrize("source_type", ["text", "markdown", "html"])
def test_ingest_explicit_source_type(client, source_type: str) -> None:
    resp = client.post(
        "/api/v1/ingest",
        json={
            "source": f"doc.{source_type}",
            "content_base64": b64(f"Content for {source_type} source."),
            "source_type": source_type,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ingested"


def test_version_endpoint(client) -> None:
    resp = client.get("/api/v1/version")
    assert resp.status_code == 200
    assert "version" in resp.json()


def test_documents_endpoint_empty(client) -> None:
    resp = client.get("/api/v1/documents")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert body["documents"] == []


def test_documents_endpoint_after_ingest(client) -> None:
    client.post(
        "/api/v1/ingest",
        json={"source": "indexed.txt", "content_base64": b64("Some indexed content here.")},
    )
    resp = client.get("/api/v1/documents")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["documents"][0]["source"] == "indexed.txt"


def test_delete_document_endpoint(client) -> None:
    client.post(
        "/api/v1/ingest",
        json={"source": "to_delete.txt", "content_base64": b64("Content to delete.")},
    )
    resp = client.delete("/api/v1/documents/to_delete.txt")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"


def test_delete_nonexistent_document(client) -> None:
    resp = client.delete("/api/v1/documents/no-such-source.txt")
    assert resp.status_code == 404


def test_correlation_id_header_passthrough(client) -> None:
    cid = "test-correlation-abc123"
    resp = client.get("/api/v1/health", headers={"X-Correlation-ID": cid})
    assert resp.headers.get("X-Correlation-ID") == cid


def test_correlation_id_generated_when_absent(client) -> None:
    resp = client.get("/api/v1/health")
    assert "X-Correlation-ID" in resp.headers
    assert len(resp.headers["X-Correlation-ID"]) > 0


@pytest.mark.parametrize(
    "question",
    [
        "What is the maximum solar rebate?",
        "How many kilowatts qualify for the program?",
        "What are the eligibility requirements?",
    ],
)
def test_query_parametrized_questions(client, question: str) -> None:
    client.post(
        "/api/v1/ingest",
        json={
            "source": "policy.txt",
            "content_base64": b64(
                "The maximum solar rebate is five thousand dollars per household. "
                "Systems up to ten kilowatts qualify for the program. "
                "Eligibility requires installation by an accredited contractor."
            ),
        },
    )
    resp = client.post("/api/v1/query", json={"question": question})
    assert resp.status_code == 200
    body = resp.json()
    assert "answered" in body
    assert "confidence" in body


def test_sources_empty(client) -> None:
    resp = client.get("/api/v1/sources")
    assert resp.status_code == 200
    body = resp.json()
    assert "sources" in body
    assert body["total"] == 0


def test_sources_after_ingest(client) -> None:
    client.post(
        "/api/v1/ingest",
        json={"source": "my-doc.txt", "content_base64": b64("Content for source listing test.")},
    )
    resp = client.get("/api/v1/sources")
    assert resp.status_code == 200
    body = resp.json()
    assert "my-doc.txt" in body["sources"]
    assert body["total"] >= 1


def test_sources_lists_all_ingested(client) -> None:
    for name in ("alpha.txt", "beta.txt", "gamma.txt"):
        client.post(
            "/api/v1/ingest",
            json={"source": name, "content_base64": b64(f"Content for {name} in index.")},
        )
    resp = client.get("/api/v1/sources")
    sources = resp.json()["sources"]
    for name in ("alpha.txt", "beta.txt", "gamma.txt"):
        assert name in sources


def test_health_endpoint_includes_config_summary(client) -> None:
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "config" in body
    assert "generator_backend" in body["config"]
    assert "final_top_k" in body["config"]
