"""Tests for veritas-rag evidence gate / hallucination fallback."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.generation.fallback import GateDecision, evaluate_evidence
from app.schemas import Chunk, ScoredChunk


def _make_chunk(chunk_id="c1", text="relevant text"):
    return Chunk(
        chunk_id=chunk_id,
        doc_id="d1",
        source="src.txt",
        source_type="text",
        page=1,
        seq=0,
        text=text,
        created_at="2026-01-01T00:00:00",
    )


def _chunk(chunk_id="c1", trust=0.8, text="relevant text"):
    return ScoredChunk(
        chunk=_make_chunk(chunk_id=chunk_id, text=text),
        trust=trust,
    )


def test_no_chunks_returns_not_allowed():
    decision = evaluate_evidence([])
    assert decision.allowed is False


def test_no_chunks_reason():
    decision = evaluate_evidence([])
    assert decision.reason == "no_chunks_retrieved"


def test_high_trust_chunk_allowed():
    chunks = [_chunk(trust=0.9)]
    decision = evaluate_evidence(chunks)
    assert decision.allowed is True


def test_low_trust_chunk_not_allowed():
    chunks = [_chunk(trust=0.1)]
    decision = evaluate_evidence(chunks)
    assert decision.allowed is False


def test_confidence_returned():
    chunks = [_chunk(trust=0.8)]
    decision = evaluate_evidence(chunks)
    assert 0.0 <= decision.confidence <= 1.0


def test_supporting_chunks_filtered():
    chunks = [_chunk("c1", trust=0.9), _chunk("c2", trust=0.1)]
    decision = evaluate_evidence(chunks, chunk_trust_floor=0.5)
    assert all(c.trust >= 0.5 for c in decision.supporting_chunks)


def test_gate_decision_has_reason():
    decision = evaluate_evidence([_chunk(trust=0.9)])
    assert isinstance(decision.reason, str)


def test_min_supporting_chunks_honored():
    chunks = [_chunk(trust=0.9)]
    decision = evaluate_evidence(chunks, min_supporting_chunks=2)
    assert decision.allowed is False


def test_gate_decision_is_dataclass():
    decision = evaluate_evidence([_chunk(trust=0.9)])
    assert isinstance(decision, GateDecision)


@pytest.mark.parametrize("trust,should_allow", [
    (0.9, True),
    (0.5, True),
    (0.1, False),
])
def test_trust_threshold_parametrized(trust, should_allow):
    chunks = [_chunk(trust=trust)]
    decision = evaluate_evidence(chunks, confidence_threshold=0.45, chunk_trust_floor=0.35)
    assert decision.allowed == should_allow
