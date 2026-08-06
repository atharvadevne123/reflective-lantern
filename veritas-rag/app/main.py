"""Veritas-Rag HTTP API."""

from __future__ import annotations

import base64
import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .config import settings
from .pipeline import RagPipeline

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Veritas-Rag",
    description=(
        "Near-zero-hallucination retrieval-augmented generation: hybrid "
        "BM25+semantic retrieval, ANN reranking, per-chunk trust scoring, "
        "constrained cited generation, and an insufficient-evidence fallback."
    ),
    version=__version__,
)

pipeline = RagPipeline()


class IngestRequest(BaseModel):
    """Payload for /ingest."""

    source: str = Field(..., min_length=1, description="File name / logical source id.")
    content_base64: str = Field(..., description="Base64-encoded file bytes.")
    source_type: str | None = Field(
        None,
        pattern="^(pdf|email|html|markdown|text)$",
        description="Override extension-based type detection.",
    )
    created_at: str | None = Field(None, description="ISO 8601 document timestamp.")


class QueryRequest(BaseModel):
    """Payload for /query."""

    question: str = Field(..., min_length=3, max_length=2000)


@app.post(f"{settings.api_prefix}/ingest", tags=["Ingestion"], summary="Ingest a document")
async def ingest(body: IngestRequest) -> dict[str, Any]:
    """Extract, normalize, dedup, version, and index one document."""
    try:
        data = base64.b64decode(body.content_base64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="content_base64 is not valid base64") from exc
    try:
        return pipeline.ingest(
            data, source=body.source, source_type=body.source_type, created_at=body.created_at
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(f"{settings.api_prefix}/query", tags=["Query"], summary="Ask a question")
async def query(body: QueryRequest) -> dict[str, Any]:
    """Answer a question with citations, or return insufficient evidence."""
    answer = pipeline.query(body.question)
    return {
        "request_id": answer.request_id,
        "answered": answer.answered,
        "confidence": answer.confidence,
        "answer": answer.text,
        "citations": [
            {
                "claim": c.claim,
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "source": c.source,
                "page": c.page,
                "quote": c.quote,
            }
            for c in answer.citations
        ],
    }


@app.get(f"{settings.api_prefix}/health", tags=["Operations"], summary="Health check")
async def health() -> dict[str, Any]:
    """Liveness plus index readiness."""
    return {
        "status": "healthy",
        "version": __version__,
        "documents": len(pipeline.versions),
        "chunks": len(pipeline.chunks),
        "generator": type(pipeline.generator).__name__,
    }


@app.get(f"{settings.api_prefix}/metrics", tags=["Operations"], summary="Operational metrics")
async def metrics() -> dict[str, Any]:
    """Index sizes, cache hit rates, and pipeline counters."""
    return pipeline.metrics()


@app.get(
    f"{settings.api_prefix}/trace/{{request_id}}",
    tags=["Observability"],
    summary="Full trace for one request",
)
async def trace(request_id: str) -> dict[str, Any]:
    """Chunk rankings, retrieval path, trust breakdowns, and attribution."""
    found = pipeline.traces.get(request_id)
    if found is None:
        raise HTTPException(status_code=404, detail="trace not found (evicted or unknown id)")
    return found.to_dict()


@app.get(f"{settings.api_prefix}/traces", tags=["Observability"], summary="Recent traces")
async def traces(limit: int = 20) -> dict[str, Any]:
    """The most recent request traces (newest last)."""
    limit = max(1, min(limit, 100))
    return {"traces": [t.to_dict() for t in pipeline.traces.recent(limit)]}


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"service": "Veritas-Rag", "version": __version__, "docs": "/docs"}
