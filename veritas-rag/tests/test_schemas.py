"""Tests for veritas-rag app/schemas.py."""

from __future__ import annotations

import pytest

from app.schemas import Answer, Citation, Chunk, Document, ScoredChunk


class TestDocument:
    def test_create_document(self) -> None:
        doc = Document(
            doc_id="doc1",
            source="test.pdf",
            source_type="pdf",
            text="hello world",
            pages=["hello world"],
            created_at="2024-01-01T00:00:00",
        )
        assert doc.doc_id == "doc1"
        assert doc.source == "test.pdf"

    def test_default_version(self) -> None:
        doc = Document(
            doc_id="d", source="s", source_type="pdf",
            text="t", pages=["t"], created_at="2024-01-01"
        )
        assert doc.version == 1

    def test_metadata_default_empty(self) -> None:
        doc = Document(
            doc_id="d", source="s", source_type="text",
            text="t", pages=["t"], created_at="2024-01-01"
        )
        assert doc.metadata == {}


class TestChunk:
    def test_create_chunk(self) -> None:
        chunk = Chunk(
            chunk_id="doc1:1:0",
            doc_id="doc1",
            source="test.pdf",
            source_type="pdf",
            page=1,
            seq=0,
            text="hello world",
            created_at="2024-01-01",
        )
        assert chunk.chunk_id == "doc1:1:0"
        assert chunk.page == 1

    def test_chunk_id_format(self) -> None:
        chunk = Chunk(
            chunk_id="doc:2:5",
            doc_id="doc",
            source="f.pdf",
            source_type="pdf",
            page=2,
            seq=5,
            text="text",
            created_at="2024-01-01",
        )
        assert ":" in chunk.chunk_id


class TestScoredChunk:
    def test_create(self) -> None:
        chunk = Chunk(
            chunk_id="d:1:0", doc_id="d", source="f", source_type="pdf",
            page=1, seq=0, text="t", created_at="2024-01-01"
        )
        sc = ScoredChunk(chunk=chunk, score=0.9, rank=1)
        assert sc.score == pytest.approx(0.9)
        assert sc.rank == 1


class TestCitation:
    def test_create(self) -> None:
        c = Citation(doc_id="d1", source="file.pdf", page=3)
        assert c.doc_id == "d1"
        assert c.page == 3


class TestAnswer:
    def test_create(self) -> None:
        a = Answer(
            answer_text="The answer is 42.",
            citations=[],
            confidence=0.8,
            request_id="r1",
        )
        assert a.confidence == pytest.approx(0.8)
        assert a.answer_text == "The answer is 42."
