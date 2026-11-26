"""Tests for veritas-rag trust scoring."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas import Chunk, ScoredChunk
from app.scoring.trust import (
    SOURCE_QUALITY_PRIORS,
    above_threshold_ratio,
    answer_confidence,
    consistency_score,
    freshness_score,
    min_trust_gate,
    normalize_trust_scores,
    score_chunk,
    source_quality_score,
    top_k_trust,
    trust_tier,
    trust_variance,
    weighted_answer_confidence,
)


def _make_scored_chunk(trust: float = 0.6, rerank_score: float = 0.5) -> ScoredChunk:
    chunk = Chunk(
        chunk_id="d1:1:0",
        doc_id="d1",
        source="test.txt",
        source_type="text",
        page=1,
        seq=0,
        text="test",
        created_at="2024-01-01",
    )
    sc = ScoredChunk(chunk=chunk)
    sc.trust = trust
    sc.rerank_score = rerank_score
    return sc


def test_freshness_score_new_doc_near_one():
    now = datetime.now(UTC)
    created = now.isoformat()
    score = freshness_score(created, now=now)
    assert score > 0.99


def test_freshness_score_old_doc_less_than_half():
    now = datetime.now(UTC)
    created = (now - timedelta(days=730)).isoformat()
    score = freshness_score(created, half_life_days=365.0, now=now)
    assert score < 0.5


def test_freshness_score_exactly_half_life():
    now = datetime.now(UTC)
    created = (now - timedelta(days=365)).isoformat()
    score = freshness_score(created, half_life_days=365.0, now=now)
    assert abs(score - 0.5) < 0.01


def test_freshness_score_invalid_date_returns_neutral():
    score = freshness_score("not-a-date")
    assert score == 0.5


def test_freshness_score_in_range():
    now = datetime.now(UTC)
    created = (now - timedelta(days=100)).isoformat()
    score = freshness_score(created, now=now)
    assert 0.0 < score <= 1.0


def test_source_quality_pdf():
    score = source_quality_score("pdf")
    assert score == SOURCE_QUALITY_PRIORS["pdf"]


def test_source_quality_email():
    score = source_quality_score("email")
    assert score == SOURCE_QUALITY_PRIORS["email"]


def test_source_quality_pdf_higher_than_email():
    assert source_quality_score("pdf") > source_quality_score("email")


def test_source_quality_unknown_type_returns_default():
    score = source_quality_score("unknown_format")
    assert 0.0 < score <= 1.0


def test_source_quality_all_known_types_in_range():
    for source_type in SOURCE_QUALITY_PRIORS:
        score = source_quality_score(source_type)
        assert 0.0 <= score <= 1.0


@pytest.mark.parametrize("source_type", ["pdf", "markdown", "text", "html", "email"])
def test_source_quality_all_types(source_type):
    score = source_quality_score(source_type)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_trust_tier_high():
    assert trust_tier(0.80) == "high"


def test_trust_tier_medium():
    assert trust_tier(0.60) == "medium"


def test_trust_tier_low():
    assert trust_tier(0.30) == "low"


def test_trust_tier_minimal():
    assert trust_tier(0.10) == "minimal"


@pytest.mark.parametrize(
    "score,expected",
    [
        (1.0, "high"),
        (0.75, "high"),
        (0.50, "medium"),
        (0.25, "low"),
        (0.0, "minimal"),
    ],
)
def test_trust_tier_boundaries(score, expected):
    assert trust_tier(score) == expected


def test_answer_confidence_empty_returns_zero():
    assert answer_confidence([]) == 0.0


def test_consistency_score_both_found_is_higher():
    score_both = consistency_score(1, 1, 0.8, 50)
    score_one = consistency_score(1, None, 0.8, 50)
    assert score_both > score_one


def test_freshness_score_monotone_decay():
    now = datetime.now(UTC)
    scores = [
        freshness_score((now - timedelta(days=d)).isoformat(), half_life_days=365.0, now=now)
        for d in [0, 100, 200, 400]
    ]
    assert scores == sorted(scores, reverse=True)


def test_weighted_answer_confidence_empty():
    assert weighted_answer_confidence([]) == 0.0


def test_weighted_answer_confidence_higher_rerank_weights_more():
    high_rerank = _make_scored_chunk(trust=0.9, rerank_score=0.9)
    low_rerank = _make_scored_chunk(trust=0.1, rerank_score=0.1)
    score = weighted_answer_confidence([high_rerank, low_rerank], rerank_weight=0.9)
    # With high rerank_weight, the high-rerank chunk dominates, pulling confidence up
    equal_weight = (0.9 + 0.1) / 2
    assert score != equal_weight


def test_weighted_answer_confidence_zero_weight_equals_mean():
    chunks = [_make_scored_chunk(trust=0.4), _make_scored_chunk(trust=0.8)]
    # At rerank_weight=0, every chunk has weight 1.0, giving simple mean
    result = weighted_answer_confidence(chunks, rerank_weight=0.0)
    expected = round((0.4 + 0.8) / 2, 6)
    assert result == expected


def test_min_trust_gate_filters_below_threshold():
    chunks = [_make_scored_chunk(trust=0.3), _make_scored_chunk(trust=0.7)]
    result = min_trust_gate(chunks, threshold=0.5)
    assert len(result) == 1
    assert result[0].trust == 0.7


def test_min_trust_gate_keeps_at_threshold():
    chunks = [_make_scored_chunk(trust=0.5)]
    result = min_trust_gate(chunks, threshold=0.5)
    assert len(result) == 1


def test_min_trust_gate_empty_input():
    assert min_trust_gate([]) == []


def test_trust_variance_single_chunk_is_zero():
    chunks = [_make_scored_chunk(trust=0.7)]
    assert trust_variance(chunks) == 0.0


def test_trust_variance_empty_is_zero():
    assert trust_variance([]) == 0.0


def test_trust_variance_identical_chunks_is_zero():
    chunks = [_make_scored_chunk(trust=0.5), _make_scored_chunk(trust=0.5)]
    assert trust_variance(chunks) == 0.0


def test_trust_variance_different_chunks_is_positive():
    chunks = [_make_scored_chunk(trust=0.2), _make_scored_chunk(trust=0.8)]
    assert trust_variance(chunks) > 0.0


def test_above_threshold_ratio_all_above():
    chunks = [_make_scored_chunk(trust=0.8), _make_scored_chunk(trust=0.9)]
    assert above_threshold_ratio(chunks, threshold=0.5) == 1.0


def test_above_threshold_ratio_none_above():
    chunks = [_make_scored_chunk(trust=0.1), _make_scored_chunk(trust=0.2)]
    assert above_threshold_ratio(chunks, threshold=0.5) == 0.0


def test_above_threshold_ratio_empty_is_zero():
    assert above_threshold_ratio([]) == 0.0


def test_top_k_trust_returns_k():
    chunks = [_make_scored_chunk(trust=float(i) / 10) for i in range(1, 6)]
    result = top_k_trust(chunks, k=3)
    assert len(result) == 3


def test_top_k_trust_sorted_descending():
    chunks = [
        _make_scored_chunk(trust=0.3),
        _make_scored_chunk(trust=0.9),
        _make_scored_chunk(trust=0.6),
    ]
    result = top_k_trust(chunks, k=3)
    assert result[0].trust >= result[1].trust >= result[2].trust


def test_normalize_trust_scores_max_becomes_one():
    chunks = [_make_scored_chunk(trust=0.4), _make_scored_chunk(trust=0.8)]
    normalize_trust_scores(chunks)
    assert max(c.trust for c in chunks) == 1.0


def test_normalize_trust_scores_empty_no_error():
    assert normalize_trust_scores([]) == []


def test_score_chunk_attaches_trust():
    sc = _make_scored_chunk(trust=0.0)
    sc.bm25_rank = 1
    sc.vector_rank = 1
    sc.rerank_score = 0.8
    sc.chunk.created_at = datetime.now(UTC).isoformat()
    sc.chunk.source_type = "pdf"
    result = score_chunk(sc)
    assert result.trust > 0.0
    assert "freshness" in result.trust_components
    assert "source_quality" in result.trust_components
    assert "consistency" in result.trust_components
