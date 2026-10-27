"""Tests for veritas-rag app/pipeline.py."""

from __future__ import annotations

import pytest

from app.pipeline import RagPipeline
from app.schemas import Document


@pytest.fixture()
def pipeline():
    return RagPipeline()


class TestRagPipelineInit:
    def test_creates_pipeline(self) -> None:
        p = RagPipeline()
        assert p is not None

    def test_has_dedup(self) -> None:
        p = RagPipeline()
        assert hasattr(p, "dedup")

    def test_has_bm25(self) -> None:
        p = RagPipeline()
        assert hasattr(p, "bm25")

    def test_has_generator(self) -> None:
        p = RagPipeline()
        assert hasattr(p, "generator")

    def test_empty_chunks_initially(self) -> None:
        p = RagPipeline()
        assert isinstance(p.chunks, dict)


class TestIngestDocument:
    def test_ingest_new_document(self, pipeline: RagPipeline) -> None:
        doc = Document(
            doc_id="d1",
            source="test.txt",
            source_type="text",
            text="The capital of France is Paris.",
            pages=["The capital of France is Paris."],
            created_at="2024-01-01",
        )
        result = pipeline.ingest_document(doc)
        assert isinstance(result, dict)

    def test_ingest_stores_chunks(self, pipeline: RagPipeline) -> None:
        doc = Document(
            doc_id="d2",
            source="doc.txt",
            source_type="text",
            text="Short document text here for testing.",
            pages=["Short document text here for testing."],
            created_at="2024-01-01",
        )
        pipeline.ingest_document(doc)
        assert len(pipeline.chunks) > 0


class TestQuery:
    def test_query_empty_corpus_returns_answer(self, pipeline: RagPipeline) -> None:
        answer = pipeline.query("What is the answer?")
        from app.schemas import Answer

        assert isinstance(answer, Answer)

    def test_query_after_ingest(self, pipeline: RagPipeline) -> None:
        doc = Document(
            doc_id="d3",
            source="test.txt",
            source_type="text",
            text="Paris is the capital of France and a major European city.",
            pages=["Paris is the capital of France and a major European city."],
            created_at="2024-01-01",
        )
        pipeline.ingest_document(doc)
        answer = pipeline.query("What is Paris?")
        from app.schemas import Answer

        assert isinstance(answer, Answer)
