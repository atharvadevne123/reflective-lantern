"""Tests for veritas-rag app/schemas.py."""

from __future__ import annotations

import pytest

from app.schemas import Answer, Chunk, ChunkList, Citation, Document, ScoredChunk, ScoredChunkList


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
            doc_id="d",
            source="s",
            source_type="pdf",
            text="t",
            pages=["t"],
            created_at="2024-01-01",
        )
        assert doc.version == 1

    def test_metadata_default_empty(self) -> None:
        doc = Document(
            doc_id="d",
            source="s",
            source_type="text",
            text="t",
            pages=["t"],
            created_at="2024-01-01",
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
            chunk_id="d:1:0",
            doc_id="d",
            source="f",
            source_type="pdf",
            page=1,
            seq=0,
            text="t",
            created_at="2024-01-01",
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


class TestTypeAliases:
    def test_chunk_list_is_list(self) -> None:
        chunks: ChunkList = []
        assert isinstance(chunks, list)

    def test_scored_chunk_list_is_list(self) -> None:
        scored: ScoredChunkList = []
        assert isinstance(scored, list)

    def test_chunk_list_accepts_chunks(self) -> None:
        chunk = Chunk(
            chunk_id="d:1:0",
            doc_id="d",
            source="f.txt",
            source_type="text",
            page=1,
            seq=0,
            text="hello",
            created_at="2024-01-01",
        )
        chunks: ChunkList = [chunk]
        assert len(chunks) == 1

    def test_scored_chunk_list_accepts_scored_chunks(self) -> None:
        chunk = Chunk(
            chunk_id="d:1:0",
            doc_id="d",
            source="f.txt",
            source_type="text",
            page=1,
            seq=0,
            text="hello",
            created_at="2024-01-01",
        )
        sc = ScoredChunk(chunk=chunk, trust=0.7)
        scored: ScoredChunkList = [sc]
        assert len(scored) == 1


class TestDocumentDefaults:
    def test_content_hash_default_empty(self) -> None:
        doc = Document(
            doc_id="d",
            source="s",
            source_type="text",
            text="t",
            pages=["t"],
            created_at="2024-01-01",
        )
        assert doc.content_hash == ""

    def test_version_default_one(self) -> None:
        doc = Document(
            doc_id="d",
            source="s",
            source_type="text",
            text="t",
            pages=["t"],
            created_at="2024-01-01",
        )
        assert doc.version == 1

    def test_metadata_mutable_default(self) -> None:
        doc1 = Document(
            doc_id="d1",
            source="s",
            source_type="text",
            text="t",
            pages=["t"],
            created_at="2024-01-01",
        )
        doc2 = Document(
            doc_id="d2",
            source="s",
            source_type="text",
            text="t",
            pages=["t"],
            created_at="2024-01-01",
        )
        doc1.metadata["key"] = "value"
        assert "key" not in doc2.metadata


class TestScoredChunkAPI:
    def test_default_scores_zero(self) -> None:
        chunk = Chunk(
            chunk_id="d:1:0",
            doc_id="d",
            source="f",
            source_type="text",
            page=1,
            seq=0,
            text="t",
            created_at="2024-01-01",
        )
        sc = ScoredChunk(chunk=chunk)
        assert sc.fused_score == 0.0
        assert sc.rerank_score == 0.0
        assert sc.trust == 0.0

    def test_repr_contains_chunk_id(self) -> None:
        chunk = Chunk(
            chunk_id="myid:1:0",
            doc_id="d",
            source="f",
            source_type="text",
            page=1,
            seq=0,
            text="t",
            created_at="2024-01-01",
        )
        sc = ScoredChunk(chunk=chunk, trust=0.5)
        assert "myid:1:0" in repr(sc)
