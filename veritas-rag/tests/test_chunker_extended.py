"""Extended tests for veritas-rag page-aware chunker."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.chunker import chunk_document
from app.schemas import Chunk, Document


def _make_doc(text: str, pages: list[str] | None = None) -> Document:
    if pages is None:
        pages = [text]
    return Document(
        doc_id="test_doc",
        source="test.txt",
        source_type="text",
        text=text,
        pages=pages,
        created_at="2026-01-01T00:00:00",
        version=1,
        content_hash="abc123",
    )


def test_chunk_document_returns_list():
    doc = _make_doc("Energy is the capacity to do work. It manifests in many forms.")
    result = chunk_document(doc)
    assert isinstance(result, list)


def test_chunk_document_empty_text_returns_empty():
    doc = _make_doc("", pages=[""])
    result = chunk_document(doc)
    assert result == []


def test_chunk_document_each_item_is_chunk():
    doc = _make_doc("Energy consumption matters. Models predict it accurately.")
    result = chunk_document(doc)
    for item in result:
        assert isinstance(item, Chunk)


def test_chunk_document_chunk_has_correct_doc_id():
    doc = _make_doc("Energy flows through circuits. Power is measured in watts.")
    result = chunk_document(doc)
    for chunk in result:
        assert chunk.doc_id == "test_doc"


def test_chunk_document_chunk_has_source():
    doc = _make_doc("Renewable energy reduces emissions.")
    result = chunk_document(doc)
    for chunk in result:
        assert chunk.source == "test.txt"


def test_chunk_document_chunk_has_non_empty_text():
    doc = _make_doc("Energy prediction is key. Models improve over time.")
    result = chunk_document(doc)
    for chunk in result:
        assert len(chunk.text.strip()) > 0


def test_chunk_document_multi_page():
    pages = [
        "First page energy content. Solar power is growing.",
        "Second page content. Wind turbines generate power.",
    ]
    doc = _make_doc(" ".join(pages), pages=pages)
    result = chunk_document(doc)
    assert len(result) >= 1


def test_chunk_document_page_numbers_are_positive():
    doc = _make_doc("Energy efficiency is vital. Buildings consume more energy than vehicles.")
    result = chunk_document(doc)
    for chunk in result:
        assert chunk.page >= 1


def test_chunk_document_seq_starts_at_zero():
    doc = _make_doc("Energy analysis. Power grids connect regions.")
    result = chunk_document(doc)
    if result:
        assert result[0].seq == 0


def test_chunk_document_small_chunk_size():
    text = "Energy is power. Power flows in circuits. Circuits consume energy."
    doc = _make_doc(text)
    result = chunk_document(doc, chunk_size=30)
    assert len(result) >= 1


@pytest.mark.parametrize("chunk_size", [200, 500, 1000])
def test_chunk_document_various_sizes(chunk_size):
    text = " ".join([f"Sentence {i} about energy consumption." for i in range(20)])
    doc = _make_doc(text)
    result = chunk_document(doc, chunk_size=chunk_size)
    assert isinstance(result, list)
    assert len(result) >= 1
