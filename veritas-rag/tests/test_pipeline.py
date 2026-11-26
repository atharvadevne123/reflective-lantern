"""Tests for veritas-rag app/pipeline.py."""

from __future__ import annotations

import pytest

from app.pipeline import RagPipeline
from app.schemas import Answer, Document


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


class TestRagPipelineIngestAPI:
    def test_ingest_text_bytes_returns_status(self) -> None:
        p = RagPipeline()
        data = b"The solar system has eight planets orbiting the sun."
        result = p.ingest(data, source="planets.txt")
        assert result["status"] in ("ingested", "empty", "unchanged", "duplicate")

    def test_ingest_new_document_status_ingested(self) -> None:
        p = RagPipeline()
        result = p.ingest(b"The moon orbits the earth every 27 days.", source="moon.txt")
        assert result["status"] == "ingested"

    def test_ingest_empty_bytes_status_empty(self) -> None:
        p = RagPipeline()
        result = p.ingest(b"", source="empty.txt")
        assert result["status"] == "empty"

    def test_ingest_same_twice_status_unchanged(self) -> None:
        p = RagPipeline()
        p.ingest(b"Jupiter is the largest planet in the solar system.", source="jupiter.txt")
        result = p.ingest(
            b"Jupiter is the largest planet in the solar system.", source="jupiter.txt"
        )
        assert result["status"] == "unchanged"

    def test_ingest_adds_chunks_to_index(self) -> None:
        p = RagPipeline()
        p.ingest(b"The earth rotates on its axis once per day.", source="earth.txt")
        assert len(p.chunks) > 0

    def test_ingest_updates_stats(self) -> None:
        p = RagPipeline()
        p.ingest(b"Venus is the second planet from the sun.", source="venus.txt")
        assert p.stats["ingested"] == 1

    def test_sources_returns_ingested_sources(self) -> None:
        p = RagPipeline()
        p.ingest(b"Mars has two moons: Phobos and Deimos.", source="mars.txt")
        assert "mars.txt" in p.sources()

    def test_document_count_increases_after_ingest(self) -> None:
        p = RagPipeline()
        before = p.document_count()
        p.ingest(b"Saturn has rings made of ice and rock.", source="saturn.txt")
        assert p.document_count() > before

    def test_metrics_returns_dict(self) -> None:
        p = RagPipeline()
        m = p.metrics()
        assert isinstance(m, dict)
        assert "documents" in m
        assert "chunks" in m
        assert "cache" in m

    def test_query_returns_answer_type(self) -> None:
        p = RagPipeline()
        p.ingest(b"Neptune is the eighth planet from the sun.", source="neptune.txt")
        answer = p.query("What is Neptune?")
        assert isinstance(answer, Answer)
        assert isinstance(answer.text, str)
        assert isinstance(answer.answered, bool)

    def test_query_batch_returns_list(self) -> None:
        p = RagPipeline()
        p.ingest(b"Uranus is an ice giant planet.", source="uranus.txt")
        answers = p.query_batch(["What is Uranus?", "What are planets?"])
        assert len(answers) == 2
        for a in answers:
            assert isinstance(a, Answer)
