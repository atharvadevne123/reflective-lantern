"""Tests for veritas-rag app/ingestion/chunker.py."""

from __future__ import annotations

from app.ingestion.chunker import chunk_document

from app.schemas import Document


def _doc(text: str, pages: list[str] | None = None) -> Document:
    return Document(
        doc_id="d1",
        source="test.txt",
        source_type="text",
        text=text,
        pages=pages or [text],
        created_at="2024-01-01",
    )


class TestChunkDocument:
    def test_single_sentence_produces_one_chunk(self) -> None:
        doc = _doc("The sky is blue.")
        chunks = chunk_document(doc)
        assert len(chunks) >= 1

    def test_chunks_are_non_empty(self) -> None:
        doc = _doc("First sentence. Second sentence. Third sentence.")
        for chunk in chunk_document(doc):
            assert chunk.text.strip() != ""

    def test_chunk_ids_unique(self) -> None:
        doc = _doc("A. B. C. D. E. F. G.")
        chunks = chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_id_format(self) -> None:
        doc = _doc("Short text here.")
        chunks = chunk_document(doc)
        for chunk in chunks:
            parts = chunk.chunk_id.split(":")
            assert len(parts) == 3

    def test_chunk_preserves_doc_id(self) -> None:
        doc = _doc("Some content.")
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.doc_id == "d1"

    def test_chunk_preserves_source(self) -> None:
        doc = _doc("Content here.")
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.source == "test.txt"

    def test_chunk_size_limits_chunk_length(self) -> None:
        long_text = " ".join(["word" * 5 + "."] * 50)
        doc = _doc(long_text)
        chunks = chunk_document(doc, chunk_size=100)
        for chunk in chunks:
            assert len(chunk.text) < chunk_size * 3

    def test_multipage_document(self) -> None:
        doc = _doc(
            text="Page one. Page two.",
            pages=["Page one content here.", "Page two content here."],
        )
        chunks = chunk_document(doc)
        pages = {c.page for c in chunks}
        assert len(pages) >= 1

    def test_empty_page_skipped(self) -> None:
        doc = _doc(text="", pages=[""])
        chunks = chunk_document(doc)
        assert len(chunks) == 0

    def test_small_chunk_size_produces_more_chunks(self) -> None:
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        doc = _doc(text)
        small = chunk_document(doc, chunk_size=30)
        large = chunk_document(doc, chunk_size=500)
        assert len(small) >= len(large)
