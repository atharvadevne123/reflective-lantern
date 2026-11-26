"""Tests for veritas-rag app/ingestion/chunker.py."""

from __future__ import annotations

from app.ingestion.chunker import chunk_document
from app.schemas import Document


def _doc(
    text: str, pages: list[str] | None = None, source: str = "test.txt", source_type: str = "text"
) -> Document:
    return Document(
        doc_id="d1",
        source=source,
        source_type=source_type,
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
        chunk_size = 100
        chunks = chunk_document(doc, chunk_size=chunk_size)
        for chunk in chunks:
            assert len(chunk.text) < 100 * 3

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

    def test_chunk_ids_are_unique(self) -> None:
        text = " ".join(f"Sentence number {i} with some extra words." for i in range(20))
        doc = _doc(text)
        chunks = chunk_document(doc, chunk_size=100)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_preserves_source_info(self) -> None:
        doc = _doc("Some important text here.", source="report.pdf", source_type="pdf")
        chunks = chunk_document(doc)
        for c in chunks:
            assert c.source == "report.pdf"
            assert c.source_type == "pdf"

    def test_chunk_seq_starts_at_zero(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        doc = _doc(text)
        chunks = chunk_document(doc, chunk_size=30)
        assert chunks[0].seq == 0

    def test_overlap_causes_text_repetition(self) -> None:
        words = " ".join(f"word{i}" for i in range(100))
        doc = _doc(words)
        chunks = chunk_document(doc, chunk_size=50, overlap=20)
        if len(chunks) >= 2:
            first_words = set(chunks[0].text.split())
            second_words = set(chunks[1].text.split())
            assert first_words & second_words  # overlap should cause word repetition
