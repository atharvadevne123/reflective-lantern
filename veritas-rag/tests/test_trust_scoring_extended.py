"""Extended trust scoring tests with parametrize and boundary coverage."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.schemas import Chunk, ScoredChunk
from app.scoring.trust import (
    SOURCE_QUALITY_PRIORS,
    above_threshold_ratio,
    answer_confidence,
    freshness_score,
    min_trust_gate,
    normalize_trust_scores,
    source_quality_score,
    top_k_trust,
    trust_variance,
)


def _make_scored_chunk(trust: float = 0.5, rerank: float = 0.5) -> ScoredChunk:
    from datetime import UTC, datetime
    chunk = Chunk(
        chunk_id="c1", doc_id="d1", source="test.txt", source_type="text",
        page=1, seq=0, text="test chunk", created_at=datetime.now(UTC).isoformat()
    )
    return ScoredChunk(chunk=chunk, rerank_score=rerank, trust=trust)


def test_freshness_score_very_new_document_near_one() -> None:
    created = datetime.now(UTC).isoformat()
    score = freshness_score(created)
    assert score > 0.95


def test_freshness_score_very_old_document_near_zero() -> None:
    created = (datetime.now(UTC) - timedelta(days=3650)).isoformat()
    score = freshness_score(created)
    assert score < 0.1


def test_freshness_score_half_life_at_exactly_half_life() -> None:
    half_life = 100.0
    created = (datetime.now(UTC) - timedelta(days=half_life)).isoformat()
    score = freshness_score(created, half_life_days=half_life)
    assert score == pytest.approx(0.5, abs=0.01)


def test_freshness_score_invalid_date_returns_neutral() -> None:
    score = freshness_score("not-a-date")
    assert score == 0.5


@pytest.mark.parametrize("half_life", [30.0, 90.0, 365.0, 730.0])
def test_freshness_score_various_half_lives(half_life: float) -> None:
    created = (datetime.now(UTC) - timedelta(days=half_life)).isoformat()
    score = freshness_score(created, half_life_days=half_life)
    assert score == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize("source_type", list(SOURCE_QUALITY_PRIORS.keys()))
def test_source_quality_score_known_sources(source_type: str) -> None:
    score = source_quality_score(source_type)
    assert score == pytest.approx(SOURCE_QUALITY_PRIORS[source_type])


def test_source_quality_score_unknown_returns_default() -> None:
    score = source_quality_score("unknown_source")
    assert 0.0 < score <= 1.0


def test_answer_confidence_empty_returns_zero() -> None:
    assert answer_confidence([]) == 0.0


def test_answer_confidence_single_chunk() -> None:
    chunk = _make_scored_chunk(trust=0.7)
    result = answer_confidence([chunk])
    assert result == pytest.approx(0.7, rel=1e-5)


def test_answer_confidence_multiple_chunks_average() -> None:
    chunks = [_make_scored_chunk(trust=0.6), _make_scored_chunk(trust=0.8)]
    result = answer_confidence(chunks)
    assert result == pytest.approx(0.7, abs=0.01)


def test_min_trust_gate_filters_below_threshold() -> None:
    chunks = [
        _make_scored_chunk(trust=0.3),
        _make_scored_chunk(trust=0.5),
        _make_scored_chunk(trust=0.7),
    ]
    result = min_trust_gate(chunks, threshold=0.4)
    assert all(c.trust >= 0.4 for c in result)
    assert len(result) == 2


def test_min_trust_gate_all_pass() -> None:
    chunks = [_make_scored_chunk(trust=0.9)]
    result = min_trust_gate(chunks, threshold=0.5)
    assert len(result) == 1


def test_min_trust_gate_none_pass() -> None:
    chunks = [_make_scored_chunk(trust=0.2)]
    result = min_trust_gate(chunks, threshold=0.5)
    assert result == []


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_top_k_trust_returns_at_most_k(k: int) -> None:
    chunks = [_make_scored_chunk(trust=float(i) / 10.0) for i in range(5)]
    result = top_k_trust(chunks, k=k)
    assert len(result) <= k


def test_top_k_trust_sorted_descending() -> None:
    chunks = [_make_scored_chunk(trust=0.3), _make_scored_chunk(trust=0.9), _make_scored_chunk(trust=0.6)]
    result = top_k_trust(chunks)
    scores = [c.trust for c in result]
    assert scores == sorted(scores, reverse=True)


def test_trust_variance_empty_returns_zero() -> None:
    assert trust_variance([]) == 0.0


def test_trust_variance_single_chunk_returns_zero() -> None:
    assert trust_variance([_make_scored_chunk(trust=0.5)]) == 0.0


def test_trust_variance_identical_trust_returns_zero() -> None:
    chunks = [_make_scored_chunk(trust=0.5)] * 3
    assert trust_variance(chunks) == pytest.approx(0.0, abs=1e-6)


def test_above_threshold_ratio_empty_returns_zero() -> None:
    assert above_threshold_ratio([]) == 0.0


@pytest.mark.parametrize("threshold,expected_ratio", [
    (0.0, 1.0),
    (0.5, 0.5),
    (1.1, 0.0),
])
def test_above_threshold_ratio_parametrized(threshold: float, expected_ratio: float) -> None:
    chunks = [_make_scored_chunk(trust=0.3), _make_scored_chunk(trust=0.7)]
    result = above_threshold_ratio(chunks, threshold=threshold)
    assert result == pytest.approx(expected_ratio, abs=0.01)


def test_normalize_trust_scores_max_becomes_one() -> None:
    chunks = [_make_scored_chunk(trust=0.4), _make_scored_chunk(trust=0.8)]
    normalize_trust_scores(chunks)
    assert max(c.trust for c in chunks) == pytest.approx(1.0)


def test_normalize_trust_scores_empty_returns_empty() -> None:
    assert normalize_trust_scores([]) == []
