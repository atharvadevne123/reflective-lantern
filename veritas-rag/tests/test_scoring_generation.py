"""Stages 4-7 tests: trust scoring, constrained generation, citations, fallback."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.generation.extractive import ExtractiveGenerator
from app.generation.fallback import evaluate_evidence
from app.generation.prompts import INSUFFICIENT_EVIDENCE_MESSAGE
from app.schemas import Chunk, ScoredChunk
from app.scoring.trust import (
    answer_confidence,
    consistency_score,
    freshness_score,
    score_chunk,
    source_quality_score,
)

NOW = datetime.now(UTC)


def make_scored(text: str = "some chunk text", **kwargs) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=kwargs.get("chunk_id", "d:1:0"),
        doc_id="d",
        source=kwargs.get("source", "doc.txt"),
        source_type=kwargs.get("source_type", "text"),
        page=1,
        seq=0,
        text=text,
        created_at=kwargs.get("created_at", NOW.isoformat()),
    )
    return ScoredChunk(
        chunk=chunk,
        bm25_rank=kwargs.get("bm25_rank"),
        vector_rank=kwargs.get("vector_rank"),
        rerank_score=kwargs.get("rerank_score", 0.0),
        trust=kwargs.get("trust", 0.0),
    )


class TestFreshness:
    def test_new_document_near_one(self):
        assert freshness_score(NOW.isoformat(), now=NOW) == pytest.approx(1.0, abs=0.01)

    def test_half_life(self):
        old = (NOW - timedelta(days=365)).isoformat()
        assert freshness_score(old, half_life_days=365, now=NOW) == pytest.approx(0.5, abs=0.01)

    def test_unparseable_timestamp_neutral(self):
        assert freshness_score("not-a-date") == 0.5


class TestSourceQuality:
    def test_pdf_above_email(self):
        assert source_quality_score("pdf") > source_quality_score("email")

    def test_unknown_type_gets_default(self):
        assert source_quality_score("carrier_pigeon") == 0.70


class TestConsistency:
    def test_both_retrievers_beats_one(self):
        both = consistency_score(1, 1, 0.5, top_k=50)
        one = consistency_score(1, None, 0.5, top_k=50)
        assert both > one

    def test_rerank_contributes(self):
        high = consistency_score(1, 1, 1.0, top_k=50)
        low = consistency_score(1, 1, 0.0, top_k=50)
        assert high > low


class TestScoreChunk:
    def test_components_recorded(self):
        scored = score_chunk(make_scored(bm25_rank=1, vector_rank=2, rerank_score=0.8), now=NOW)
        assert set(scored.trust_components) == {"freshness", "source_quality", "consistency"}
        assert 0.0 <= scored.trust <= 1.0

    def test_fresh_agreeing_chunk_trusted(self):
        scored = score_chunk(
            make_scored(bm25_rank=1, vector_rank=1, rerank_score=0.9, source_type="pdf"),
            now=NOW,
        )
        assert scored.trust > 0.7

    def test_stale_single_retriever_chunk_less_trusted(self):
        stale = (NOW - timedelta(days=3000)).isoformat()
        scored = score_chunk(
            make_scored(
                bm25_rank=None,
                vector_rank=40,
                rerank_score=0.1,
                created_at=stale,
                source_type="email",
            ),
            now=NOW,
        )
        assert scored.trust < 0.4


class TestEvidenceGate:
    def test_no_chunks_blocks(self):
        decision = evaluate_evidence([])
        assert not decision.allowed
        assert decision.reason == "no_chunks_retrieved"

    def test_low_trust_blocks(self):
        chunks = [make_scored(trust=0.1), make_scored(trust=0.2)]
        decision = evaluate_evidence(chunks, chunk_trust_floor=0.35)
        assert not decision.allowed
        assert decision.reason == "too_few_supporting_chunks"

    def test_confidence_threshold_blocks(self):
        chunks = [make_scored(trust=0.40)]
        decision = evaluate_evidence(chunks, confidence_threshold=0.60, chunk_trust_floor=0.35)
        assert not decision.allowed
        assert decision.reason == "confidence_below_threshold"

    def test_strong_evidence_allows(self):
        chunks = [make_scored(trust=0.8), make_scored(trust=0.7)]
        decision = evaluate_evidence(chunks)
        assert decision.allowed
        assert decision.confidence == pytest.approx(0.75)

    def test_answer_confidence_empty(self):
        assert answer_confidence([]) == 0.0


class TestExtractiveGenerator:
    def test_answer_sentences_verbatim_from_evidence(self):
        gen = ExtractiveGenerator()
        chunk = make_scored(
            "The rebate pays 900 dollars per kilowatt. Applications close in June.",
            trust=0.8,
        )
        answer, citations = gen.generate("How much does the rebate pay per kilowatt?", [chunk])
        assert "900 dollars" in answer
        for cite in citations:
            assert cite.quote in chunk.chunk.text  # zero-hallucination property

    def test_every_claim_cited(self):
        gen = ExtractiveGenerator()
        chunk = make_scored("Solar exports are capped at five kilowatts.", trust=0.9)
        answer, citations = gen.generate("What is the solar export cap?", [chunk])
        assert answer
        assert len(citations) >= 1
        assert citations[0].page == 1
        assert citations[0].source == "doc.txt"

    def test_irrelevant_evidence_yields_empty(self):
        gen = ExtractiveGenerator()
        chunk = make_scored("Bananas are yellow fruit.", trust=0.9)
        answer, citations = gen.generate("What is the tax rate on diesel?", [chunk])
        assert answer == ""
        assert citations == []


class TestPipelineEndToEnd:
    def test_answer_carries_citations_with_pages(self, pipeline):
        answer = pipeline.query("What is the maximum rebate per household?")
        assert answer.answered
        assert answer.citations
        for cite in answer.citations:
            assert cite.doc_id
            assert cite.page >= 1
            assert cite.quote

    def test_unanswerable_query_falls_back(self, pipeline):
        answer = pipeline.query("What is the average rainfall in the Amazon basin?")
        assert not answer.answered
        assert answer.text == INSUFFICIENT_EVIDENCE_MESSAGE
        assert answer.citations == []

    def test_empty_corpus_falls_back(self, empty_pipeline):
        answer = empty_pipeline.query("Anything at all?")
        assert not answer.answered
        assert answer.text == INSUFFICIENT_EVIDENCE_MESSAGE

    def test_confidence_reported(self, pipeline):
        answer = pipeline.query("How much is the rooftop solar rebate per kilowatt?")
        assert answer.answered
        assert 0.0 < answer.confidence <= 1.0


class TestTrustTier:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (0.80, "high"),
            (0.75, "high"),
            (0.74, "medium"),
            (0.50, "medium"),
            (0.49, "low"),
            (0.25, "low"),
            (0.24, "minimal"),
            (0.0, "minimal"),
        ],
    )
    def test_tier_boundaries(self, score, expected):
        from app.scoring.trust import trust_tier

        assert trust_tier(score) == expected


class TestWeightedAnswerConfidence:
    def test_empty_returns_zero(self):
        from app.scoring.trust import weighted_answer_confidence

        assert weighted_answer_confidence([]) == pytest.approx(0.0)

    def test_single_chunk(self):
        from app.scoring.trust import weighted_answer_confidence

        chunk = make_scored(trust=0.8, rerank_score=1.0)
        result = weighted_answer_confidence([chunk])
        assert 0.0 <= result <= 1.0

    def test_higher_rerank_weighs_more(self):
        from app.scoring.trust import weighted_answer_confidence

        low_rerank = make_scored(trust=0.5, rerank_score=0.1)
        high_rerank = make_scored(trust=0.9, rerank_score=0.9)
        result = weighted_answer_confidence([low_rerank, high_rerank])
        assert result > 0.5


class TestMinTrustGate:
    def test_empty_returns_empty(self):
        from app.scoring.trust import min_trust_gate

        assert min_trust_gate([]) == []

    def test_all_above_threshold_pass(self):
        from app.scoring.trust import min_trust_gate

        chunks = [make_scored(trust=0.8), make_scored(trust=0.9)]
        assert len(min_trust_gate(chunks, threshold=0.5)) == 2

    def test_below_threshold_filtered(self):
        from app.scoring.trust import min_trust_gate

        chunks = [make_scored(trust=0.2), make_scored(trust=0.8)]
        result = min_trust_gate(chunks, threshold=0.4)
        assert len(result) == 1
        assert result[0].trust == pytest.approx(0.8)

    @pytest.mark.parametrize(
        "trusts,threshold,expected_count",
        [
            ([0.1, 0.5, 0.9], 0.4, 2),
            ([0.1, 0.2, 0.3], 0.4, 0),
            ([0.7, 0.8, 0.9], 0.4, 3),
        ],
    )
    def test_parametrized_gate(self, trusts, threshold, expected_count):
        from app.scoring.trust import min_trust_gate

        chunks = [make_scored(trust=t) for t in trusts]
        assert len(min_trust_gate(chunks, threshold=threshold)) == expected_count


class TestTrustVariance:
    def test_zero_variance_single_chunk(self) -> None:
        from app.scoring.trust import trust_variance

        assert trust_variance([make_scored(trust=0.8)]) == 0.0

    def test_zero_variance_identical(self) -> None:
        from app.scoring.trust import trust_variance

        chunks = [make_scored(trust=0.5), make_scored(trust=0.5)]
        assert trust_variance(chunks) == pytest.approx(0.0, abs=1e-6)

    def test_known_variance(self) -> None:
        from app.scoring.trust import trust_variance

        chunks = [make_scored(trust=0.0), make_scored(trust=1.0)]
        assert trust_variance(chunks) == pytest.approx(0.25, abs=1e-4)

    def test_empty_returns_zero(self) -> None:
        from app.scoring.trust import trust_variance

        assert trust_variance([]) == 0.0


class TestAboveThresholdRatio:
    def test_all_above(self) -> None:
        from app.scoring.trust import above_threshold_ratio

        chunks = [make_scored(trust=0.8), make_scored(trust=0.9)]
        assert above_threshold_ratio(chunks, threshold=0.5) == pytest.approx(1.0)

    def test_none_above(self) -> None:
        from app.scoring.trust import above_threshold_ratio

        chunks = [make_scored(trust=0.2), make_scored(trust=0.3)]
        assert above_threshold_ratio(chunks, threshold=0.5) == pytest.approx(0.0)

    def test_half_above(self) -> None:
        from app.scoring.trust import above_threshold_ratio

        chunks = [make_scored(trust=0.4), make_scored(trust=0.6)]
        assert above_threshold_ratio(chunks, threshold=0.5) == pytest.approx(0.5)

    def test_empty_returns_zero(self) -> None:
        from app.scoring.trust import above_threshold_ratio

        assert above_threshold_ratio([]) == 0.0


class TestTopKTrust:
    def test_returns_top_k(self) -> None:
        from app.scoring.trust import top_k_trust

        chunks = [make_scored(trust=0.3), make_scored(trust=0.9), make_scored(trust=0.6)]
        result = top_k_trust(chunks, k=2)
        assert len(result) == 2
        assert result[0].trust == pytest.approx(0.9)

    def test_fewer_than_k_returns_all(self) -> None:
        from app.scoring.trust import top_k_trust

        chunks = [make_scored(trust=0.5)]
        assert len(top_k_trust(chunks, k=3)) == 1

    def test_empty_returns_empty(self) -> None:
        from app.scoring.trust import top_k_trust

        assert top_k_trust([], k=5) == []
