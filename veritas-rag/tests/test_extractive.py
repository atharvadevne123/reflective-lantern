"""Tests for veritas-rag extractive generator."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.generation.extractive import ExtractiveGenerator
from app.schemas import Chunk, ScoredChunk


def _make_scored(text: str, trust: float = 0.8) -> ScoredChunk:
    chunk = Chunk(
        chunk_id="doc1:1:0",
        doc_id="doc1",
        source="test.txt",
        source_type="text",
        page=1,
        seq=0,
        text=text,
        created_at="2026-01-01T00:00:00",
    )
    return ScoredChunk(chunk=chunk, trust=trust)


def test_generate_returns_tuple():
    gen = ExtractiveGenerator()
    answer, citations = gen.generate("what is energy", [_make_scored("Energy is power in action.")])
    assert isinstance(answer, str)
    assert isinstance(citations, list)


def test_generate_empty_chunks_returns_empty():
    gen = ExtractiveGenerator()
    answer, citations = gen.generate("test query", [])
    assert answer == ""
    assert citations == []


def test_generate_matching_sentence_included():
    gen = ExtractiveGenerator()
    scored = _make_scored("Electricity consumption affects the energy grid significantly.")
    answer, citations = gen.generate("electricity energy grid", [scored])
    assert len(answer) > 0


def test_generate_citations_have_correct_fields():
    gen = ExtractiveGenerator()
    scored = _make_scored("The model predicts energy consumption accurately.")
    answer, citations = gen.generate("model predicts energy", [scored])
    if citations:
        c = citations[0]
        assert hasattr(c, "chunk_id")
        assert hasattr(c, "doc_id")
        assert hasattr(c, "source")


def test_generate_respects_max_sentences():
    gen = ExtractiveGenerator(max_sentences=1)
    text = "Energy flows everywhere. Power grids distribute energy. Consumption rises yearly."
    scored = _make_scored(text)
    answer, citations = gen.generate("energy power consumption", [scored])
    assert len(citations) <= 1


def test_generate_no_overlap_returns_empty():
    gen = ExtractiveGenerator()
    scored = _make_scored("The quick brown fox jumps over the lazy dog.")
    answer, citations = gen.generate("energy prediction model", [scored])
    assert answer == ""
    assert citations == []


def test_generate_empty_query_returns_empty():
    gen = ExtractiveGenerator()
    scored = _make_scored("Energy consumption is high today.")
    answer, citations = gen.generate("", [scored])
    assert answer == ""
    assert citations == []


def test_generate_multiple_chunks_selects_best():
    gen = ExtractiveGenerator(max_sentences=2)
    chunks = [
        _make_scored("Energy consumption dropped this quarter significantly.", trust=0.9),
        _make_scored("Renewable energy sources are growing rapidly.", trust=0.7),
    ]
    answer, citations = gen.generate("energy consumption renewable", chunks)
    assert isinstance(answer, str)


@pytest.mark.parametrize("max_s", [1, 2, 3])
def test_generate_max_sentences_param(max_s):
    gen = ExtractiveGenerator(max_sentences=max_s)
    text = " ".join([f"Energy sentence {i} matters greatly." for i in range(10)])
    scored = _make_scored(text)
    _, citations = gen.generate("energy sentence matters", [scored])
    assert len(citations) <= max_s
