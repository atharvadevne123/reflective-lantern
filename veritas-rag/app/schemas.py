"""Shared dataclasses used across the Veritas-Rag pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A normalized source document after extraction.

    Attributes:
        doc_id: Stable identifier (content-hash derived).
        source: Original file name or logical source name.
        source_type: One of pdf / email / text / html / markdown.
        text: Full normalized text.
        pages: Per-page text (single entry for non-paginated sources).
        created_at: ISO 8601 timestamp of document creation/modification.
        version: Monotonic version number within its version chain.
        content_hash: SHA-256 of normalized text.
        metadata: Arbitrary source metadata (email headers, PDF info, ...).
    """

    doc_id: str
    source: str
    source_type: str
    text: str
    pages: list[str]
    created_at: str
    version: int = 1
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable unit of text with provenance.

    Attributes:
        chunk_id: Unique id: "<doc_id>:<page>:<seq>".
        doc_id: Parent document id.
        source: Parent document source name.
        source_type: Parent document source type.
        page: 1-indexed page number the chunk starts on.
        seq: Sequence number within the document.
        text: Chunk text.
        created_at: Parent document timestamp (for freshness scoring).
    """

    chunk_id: str
    doc_id: str
    source: str
    source_type: str
    page: int
    seq: int
    text: str
    created_at: str


@dataclass
class ScoredChunk:
    """A chunk with retrieval and trust scores attached.

    Attributes:
        chunk: The underlying chunk.
        bm25_rank: Rank in the BM25 result list (None if absent).
        vector_rank: Rank in the vector result list (None if absent).
        fused_score: Reciprocal-rank-fusion score.
        rerank_score: Deep scorer output in [0, 1].
        trust: Composite trust score in [0, 1].
        trust_components: Breakdown: freshness / source_quality / consistency.
    """

    chunk: Chunk
    bm25_rank: int | None = None
    vector_rank: int | None = None
    fused_score: float = 0.0
    rerank_score: float = 0.0
    trust: float = 0.0
    trust_components: dict[str, float] = field(default_factory=dict)


@dataclass
class Citation:
    """A claim-to-evidence link in a generated answer.

    Attributes:
        claim: The answer sentence being supported.
        chunk_id: Supporting chunk id.
        doc_id: Supporting document id.
        source: Document source name.
        page: Page number of the evidence.
        quote: The supporting text span.
    """

    claim: str
    chunk_id: str
    doc_id: str
    source: str
    page: int
    quote: str


@dataclass
class Answer:
    """Final pipeline output.

    Attributes:
        text: Answer text, or the insufficient-evidence message.
        answered: False when the fallback fired.
        confidence: Mean trust of the supporting chunks.
        citations: Claim-level citations.
        request_id: Trace identifier for observability lookups.
    """

    text: str
    answered: bool
    confidence: float
    citations: list[Citation]
    request_id: str
