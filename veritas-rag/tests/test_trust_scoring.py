"""Focused tests for app/scoring/trust.py.

Covers each trust component in isolation and the weighted composite, using
an injected clock so freshness results are deterministic.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.schemas import Chunk, ScoredChunk
from app.scoring.trust import (
    SOURCE_QUALITY_PRIORS,
    answer_confidence,
    consistency_score,
    freshness_score,
    score_chunk,
    source_quality_score,
    trust_tier,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def make_chunk(
    created_at: str | None = None,
    source_type: str = "markdown",
) -> Chunk:
    """Build a Chunk with sensible defaults for scoring tests."""
    return Chunk(
        chunk_id="doc-1:1:0",
        doc_id="doc-1",
        source="handbook.md",
        source_type=source_type,
        page=1,
        seq=0,
        text="Rooftop solar rebates are paid per installed kilowatt.",
        created_at=created_at if created_at is not None else NOW.isoformat(),
    )


def make_scored(**overrides) -> ScoredChunk:
    """Build a ScoredChunk with retrieval signals overridable per test."""
    params = {
        "chunk": make_chunk(),
        "bm25_rank": 1,
        "vector_rank": 1,
        "rerank_score": 1.0,
    }
    params.update(overrides)
    return ScoredChunk(**params)


class TestFreshnessScore:
    def test_brand_new_document_scores_near_one(self) -> None:
        assert freshness_score(NOW.isoformat(), now=NOW) == pytest.approx(1.0)

    def test_half_life_halves_the_score(self) -> None:
        created = (NOW - timedelta(days=365)).isoformat()
        assert freshness_score(created, half_life_days=365.0, now=NOW) == pytest.approx(
            0.5, rel=1e-6
        )

    def test_two_half_lives_quarter_the_score(self) -> None:
        created = (NOW - timedelta(days=730)).isoformat()
        assert freshness_score(created, half_life_days=365.0, now=NOW) == pytest.approx(
            0.25, rel=1e-6
        )

    def test_older_documents_score_lower(self) -> None:
        recent = freshness_score((NOW - timedelta(days=10)).isoformat(), now=NOW)
        stale = freshness_score((NOW - timedelta(days=1000)).isoformat(), now=NOW)
        assert recent > stale

    def test_shorter_half_life_decays_faster(self) -> None:
        created = (NOW - timedelta(days=100)).isoformat()
        assert freshness_score(created, half_life_days=30.0, now=NOW) < freshness_score(
            created, half_life_days=365.0, now=NOW
        )

    def test_future_timestamp_clamps_to_one(self) -> None:
        created = (NOW + timedelta(days=30)).isoformat()
        assert freshness_score(created, now=NOW) == pytest.approx(1.0)

    def test_naive_timestamp_is_treated_as_utc(self) -> None:
        naive = NOW.replace(tzinfo=None).isoformat()
        assert freshness_score(naive, now=NOW) == pytest.approx(1.0)

    def test_unparseable_timestamp_is_neutral(self) -> None:
        assert freshness_score("not-a-date", now=NOW) == 0.5

    def test_score_always_within_unit_interval(self) -> None:
        for days in (0, 1, 100, 5000):
            score = freshness_score((NOW - timedelta(days=days)).isoformat(), now=NOW)
            assert 0.0 < score <= 1.0


class TestSourceQualityScore:
    @pytest.mark.parametrize("source_type", sorted(SOURCE_QUALITY_PRIORS))
    def test_known_types_return_their_prior(self, source_type: str) -> None:
        assert source_quality_score(source_type) == SOURCE_QUALITY_PRIORS[source_type]

    def test_unknown_type_falls_back_to_default(self) -> None:
        assert source_quality_score("parquet") == 0.70

    def test_authored_formats_outrank_scraped_ones(self) -> None:
        assert source_quality_score("pdf") > source_quality_score("html")

    def test_curated_docs_outrank_informal_email(self) -> None:
        assert source_quality_score("markdown") > source_quality_score("email")

    def test_all_priors_are_valid_probabilities(self) -> None:
        assert all(0.0 <= v <= 1.0 for v in SOURCE_QUALITY_PRIORS.values())


class TestConsistencyScore:
    def test_top_rank_in_both_retrievers_scores_highest(self) -> None:
        assert consistency_score(1, 1, 1.0, top_k=50) == pytest.approx(1.0)

    def test_found_by_neither_retriever_scores_lowest(self) -> None:
        assert consistency_score(None, None, 0.0, top_k=50) == pytest.approx(0.0)

    def test_agreement_beats_a_single_retriever(self) -> None:
        both = consistency_score(1, 1, 0.5, top_k=50)
        one = consistency_score(1, None, 0.5, top_k=50)
        assert both > one

    def test_better_rank_scores_higher(self) -> None:
        assert consistency_score(1, 1, 0.5, top_k=50) > consistency_score(40, 40, 0.5, top_k=50)

    def test_rerank_score_lifts_the_result(self) -> None:
        assert consistency_score(10, 10, 1.0, top_k=50) > consistency_score(10, 10, 0.0, top_k=50)

    def test_rerank_score_above_one_is_capped(self) -> None:
        assert consistency_score(1, 1, 5.0, top_k=50) == consistency_score(1, 1, 1.0, top_k=50)

    def test_rank_beyond_top_k_earns_no_rank_credit(self) -> None:
        # A rank past the window contributes nothing beyond the both-found bonus.
        assert consistency_score(100, 100, 0.0, top_k=10) == pytest.approx(0.35)

    def test_score_always_within_unit_interval(self) -> None:
        for bm25, vector, rerank in ((1, 1, 1.0), (None, 5, 0.3), (None, None, 0.0), (50, 1, 0.9)):
            assert 0.0 <= consistency_score(bm25, vector, rerank, top_k=50) <= 1.0


class TestScoreChunk:
    def test_attaches_trust_and_components(self) -> None:
        result = score_chunk(make_scored(), now=NOW)
        assert 0.0 <= result.trust <= 1.0
        assert set(result.trust_components) == {"freshness", "source_quality", "consistency"}

    def test_best_case_chunk_scores_near_one(self) -> None:
        scored = make_scored(chunk=make_chunk(source_type="pdf"))
        assert score_chunk(scored, now=NOW).trust > 0.9

    def test_stale_single_retriever_chunk_scores_low(self) -> None:
        scored = make_scored(
            chunk=make_chunk(
                created_at=(NOW - timedelta(days=3650)).isoformat(), source_type="email"
            ),
            bm25_rank=None,
            vector_rank=45,
            rerank_score=0.1,
        )
        assert score_chunk(scored, now=NOW).trust < 0.4

    def test_composite_matches_weighted_components(self) -> None:
        result = score_chunk(make_scored(), now=NOW)
        components = result.trust_components
        expected = (
            0.30 * components["freshness"]
            + 0.25 * components["source_quality"]
            + 0.45 * components["consistency"]
        )
        assert result.trust == pytest.approx(expected, abs=1e-5)

    def test_freshness_drives_trust_down_as_age_grows(self) -> None:
        fresh = score_chunk(make_scored(), now=NOW).trust
        stale = score_chunk(
            make_scored(chunk=make_chunk(created_at=(NOW - timedelta(days=2000)).isoformat())),
            now=NOW,
        ).trust
        assert fresh > stale

    def test_returns_the_same_object(self) -> None:
        scored = make_scored()
        assert score_chunk(scored, now=NOW) is scored


class TestAnswerConfidence:
    def test_empty_set_has_no_confidence(self) -> None:
        assert answer_confidence([]) == 0.0

    def test_single_chunk_confidence_is_its_trust(self) -> None:
        scored = score_chunk(make_scored(), now=NOW)
        assert answer_confidence([scored]) == pytest.approx(scored.trust)

    def test_is_the_mean_of_supporting_trust(self) -> None:
        high = make_scored()
        high.trust = 0.9
        low = make_scored()
        low.trust = 0.5
        assert answer_confidence([high, low]) == pytest.approx(0.7)

    def test_a_weak_chunk_drags_confidence_down(self) -> None:
        strong = make_scored()
        strong.trust = 0.9
        weak = make_scored()
        weak.trust = 0.1
        assert answer_confidence([strong, weak]) < answer_confidence([strong])


class TestTrustTier:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (1.0, "high"),
            (0.75, "high"),
            (0.74, "medium"),
            (0.50, "medium"),
            (0.49, "low"),
            (0.25, "low"),
            (0.24, "minimal"),
            (0.0, "minimal"),
        ],
    )
    def test_tier_boundaries(self, score: float, expected: str) -> None:
        assert trust_tier(score) == expected

    def test_tier_is_monotonic_in_score(self) -> None:
        order = {"minimal": 0, "low": 1, "medium": 2, "high": 3}
        tiers = [order[trust_tier(s / 10)] for s in range(11)]
        assert tiers == sorted(tiers)
