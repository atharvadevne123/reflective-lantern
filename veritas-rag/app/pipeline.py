"""The Veritas-Rag pipeline: all ten stages wired together."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from .caching import RetrievalCache
from .config import Settings
from .config import settings as default_settings
from .generation.extractive import ExtractiveGenerator
from .generation.fallback import evaluate_evidence
from .generation.prompts import INSUFFICIENT_EVIDENCE_MESSAGE
from .ingestion.chunker import chunk_document
from .ingestion.dedup import DuplicateDetector, content_hash
from .ingestion.extractors import detect_source_type, extract_text
from .ingestion.normalize import normalize_text, strip_email_quotes
from .ingestion.versioning import VersionStore, make_doc_id
from .observability import TraceRecorder, TraceStore
from .retrieval.ann import VectorIndex
from .retrieval.bm25 import BM25Index
from .retrieval.embedder import HashingEmbedder
from .retrieval.hybrid import reciprocal_rank_fusion
from .retrieval.rerank import LexicalReranker
from .schemas import Answer, Chunk, Document, ScoredChunk
from .scoring.trust import score_chunk

logger = logging.getLogger(__name__)


class RagPipeline:
    """In-process implementation of the ten-stage architecture."""

    def __init__(self, config: Settings | None = None, generator: Any | None = None) -> None:
        self.config = config or default_settings

        # Stage 1 state
        self.dedup = DuplicateDetector(near_threshold=self.config.near_dup_threshold)
        self.versions = VersionStore()
        self.chunks: dict[str, Chunk] = {}

        # Stage 2-3 state
        self.bm25 = BM25Index()
        self.embedder = HashingEmbedder(dim=self.config.embedding_dim)
        self.vectors = VectorIndex(dim=self.config.embedding_dim)
        self.reranker = LexicalReranker()

        # Stage 5-6
        if generator is not None:
            self.generator = generator
        elif self.config.generator_backend == "anthropic":
            from .generation.llm import AnthropicGenerator

            self.generator = AnthropicGenerator(model=self.config.anthropic_model)
        else:
            self.generator = ExtractiveGenerator()

        # Stage 9-10
        self.cache = RetrievalCache(
            max_size=self.config.cache_max_size,
            ttl_seconds=self.config.cache_ttl_seconds,
        )
        self.traces = TraceStore(max_size=self.config.trace_buffer_size)

        self.stats = {"ingested": 0, "duplicates_skipped": 0, "queries": 0}

    # ------------------------------------------------------------------
    # Stage 1 — ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        data: bytes,
        source: str,
        source_type: str | None = None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        """Ingest one document: extract, normalize, dedup, version, index.

        Args:
            data: Raw file bytes.
            source: Logical source name (file name).
            source_type: Override the extension-based type detection.
            created_at: ISO 8601 timestamp; defaults to now.

        Returns:
            Ingest report: status, doc_id, version, chunk count.
        """
        stype = source_type or detect_source_type(source)
        pages, metadata = extract_text(data, stype)

        pages = [normalize_text(p) for p in pages]
        if stype == "email":
            pages = [strip_email_quotes(p) for p in pages]
        full_text = "\n\n".join(p for p in pages if p)

        if not full_text.strip():
            return {"status": "empty", "source": source}

        digest = content_hash(full_text)

        # Re-ingest of identical content for the same source: no-op
        if self.versions.is_unchanged(source, digest):
            latest = self.versions.latest(source)
            return {
                "status": "unchanged",
                "doc_id": latest.doc_id if latest else None,
                "source": source,
            }

        # Cross-source duplicate detection (exact + near-dup)
        duplicate_of, similarity = self.dedup.check(full_text)
        is_new_version_of_same_source = self.versions.latest(source) is not None
        if duplicate_of is not None and not is_new_version_of_same_source:
            self.stats["duplicates_skipped"] += 1
            return {
                "status": "duplicate",
                "duplicate_of": duplicate_of,
                "similarity": round(similarity, 4),
                "source": source,
            }

        version = self.versions.next_version(source)
        doc = Document(
            doc_id=make_doc_id(source, version),
            source=source,
            source_type=stype,
            text=full_text,
            pages=pages,
            created_at=created_at or datetime.now(UTC).isoformat(),
            version=version,
            content_hash=digest,
            metadata=metadata,
        )

        # Retire superseded versions from the live indexes
        for old in self.versions.history(source):
            self._remove_document_chunks(old.doc_id)
        self.versions.add(doc)
        self.dedup.register(doc.doc_id, full_text)

        new_chunks = chunk_document(
            doc, chunk_size=self.config.chunk_size, overlap=self.config.chunk_overlap
        )
        self._index_chunks(new_chunks)

        # Any mutation invalidates cached retrievals
        self.cache.invalidate_all()
        self.stats["ingested"] += 1

        return {
            "status": "ingested",
            "doc_id": doc.doc_id,
            "version": doc.version,
            "chunks": len(new_chunks),
            "source": source,
        }

    def _index_chunks(self, new_chunks: list[Chunk]) -> None:
        if not new_chunks:
            return
        for chunk in new_chunks:
            self.chunks[chunk.chunk_id] = chunk
            self.bm25.add(chunk.chunk_id, chunk.text)
        vectors = self.embedder.embed([c.text for c in new_chunks])
        self.vectors.add([c.chunk_id for c in new_chunks], vectors)

    def _remove_document_chunks(self, doc_id: str) -> None:
        stale = [cid for cid, chunk in self.chunks.items() if chunk.doc_id == doc_id]
        for cid in stale:
            del self.chunks[cid]
            self.bm25.remove(cid)
        self.vectors.remove(stale)

    # ------------------------------------------------------------------
    # Stages 2-10 — query
    # ------------------------------------------------------------------

    def query(self, question: str) -> Answer:
        """Answer a question through the full retrieval + generation stack."""
        self.stats["queries"] += 1
        recorder = TraceRecorder(question)

        # Stage 9 — cache
        recorder.start_stage("cache")
        cached = self.cache.get(question)
        recorder.end_stage("cache", hit=cached is not None)

        if cached is not None:
            scored_chunks: list[ScoredChunk] = cached
        else:
            scored_chunks = self._retrieve(question, recorder)
            self.cache.put(question, scored_chunks)

        # Stage 7 — evidence gate (before generation: don't pay for answers
        # the gate would reject anyway)
        recorder.start_stage("evidence_gate")
        decision = evaluate_evidence(
            scored_chunks,
            confidence_threshold=self.config.confidence_threshold,
            chunk_trust_floor=self.config.chunk_trust_floor,
            min_supporting_chunks=self.config.min_supporting_chunks,
        )
        recorder.end_stage(
            "evidence_gate",
            allowed=decision.allowed,
            confidence=decision.confidence,
            reason=decision.reason,
            supporting=[c.chunk.chunk_id for c in decision.supporting_chunks],
        )

        if not decision.allowed:
            trace = recorder.finish(
                answered=False, confidence=decision.confidence, reason=decision.reason
            )
            self.traces.add(trace)
            return Answer(
                text=INSUFFICIENT_EVIDENCE_MESSAGE,
                answered=False,
                confidence=decision.confidence,
                citations=[],
                request_id=trace.request_id,
            )

        # Stages 5-6 — constrained generation with citations
        recorder.start_stage("generation")
        answer_text, citations = self.generator.generate(question, decision.supporting_chunks)
        recorder.end_stage(
            "generation",
            backend=type(self.generator).__name__,
            citations=len(citations),
            # Token attribution: which chunks the answer text draws on
            attribution=[
                {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "page": c.page} for c in citations
            ],
        )

        # A generator that produced nothing grounded also falls back
        if not answer_text.strip():
            trace = recorder.finish(
                answered=False,
                confidence=decision.confidence,
                reason="generator_found_no_grounded_answer",
            )
            self.traces.add(trace)
            return Answer(
                text=INSUFFICIENT_EVIDENCE_MESSAGE,
                answered=False,
                confidence=decision.confidence,
                citations=[],
                request_id=trace.request_id,
            )

        trace = recorder.finish(answered=True, confidence=decision.confidence)
        self.traces.add(trace)
        return Answer(
            text=answer_text,
            answered=True,
            confidence=decision.confidence,
            citations=citations,
            request_id=trace.request_id,
        )

    def _retrieve(self, question: str, recorder: TraceRecorder) -> list[ScoredChunk]:
        """Stages 2-4: hybrid retrieval, fusion, rerank, trust scoring."""
        # Stage 2a — keyword matching
        recorder.start_stage("bm25")
        bm25_results = self.bm25.search(question, top_k=self.config.bm25_top_k)
        recorder.end_stage("bm25", top=[c for c, _ in bm25_results[:10]])

        # Stage 2b / 3a — semantic ANN
        recorder.start_stage("vector")
        query_vec = self.embedder.embed([question])[0]
        vector_results = self.vectors.search(query_vec, top_k=self.config.vector_top_k)
        recorder.end_stage("vector", top=[c for c, _ in vector_results[:10]])

        # Stage 2c — fusion
        recorder.start_stage("fusion")
        fused = reciprocal_rank_fusion(bm25_results, vector_results)
        fused = fused[: self.config.rerank_top_k]
        recorder.end_stage("fusion", candidates=[c for c, _, _, _ in fused])

        candidates = [
            ScoredChunk(
                chunk=self.chunks[cid],
                bm25_rank=b_rank,
                vector_rank=v_rank,
                fused_score=round(score, 6),
            )
            for cid, score, b_rank, v_rank in fused
            if cid in self.chunks
        ]

        # Stage 3b — deep scoring
        recorder.start_stage("rerank")
        rerank_scores = self.reranker.score(question, [c.chunk.text for c in candidates])
        for scored, r in zip(candidates, rerank_scores, strict=True):
            scored.rerank_score = r
        candidates.sort(key=lambda c: c.rerank_score, reverse=True)
        candidates = candidates[: self.config.final_top_k]
        recorder.end_stage(
            "rerank",
            ranking=[{"chunk_id": c.chunk.chunk_id, "score": c.rerank_score} for c in candidates],
        )

        # Stage 4 — trust scoring
        recorder.start_stage("trust")
        for scored in candidates:
            score_chunk(
                scored,
                half_life_days=self.config.freshness_half_life_days,
                top_k=self.config.bm25_top_k,
            )
        recorder.end_stage(
            "trust",
            scores=[
                {
                    "chunk_id": c.chunk.chunk_id,
                    "trust": c.trust,
                    "components": c.trust_components,
                }
                for c in candidates
            ],
        )
        return candidates

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def metrics(self) -> dict[str, Any]:
        """Operational counters for the /metrics endpoint."""
        return {
            "documents": len(self.versions),
            "chunks": len(self.chunks),
            "bm25_index_size": len(self.bm25),
            "vector_index_size": len(self.vectors),
            "cache": self.cache.stats(),
            "traces_retained": len(self.traces),
            **self.stats,
        }
