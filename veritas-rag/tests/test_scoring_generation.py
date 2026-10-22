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
    def test_new_document_near_one(self) -> None:
        assert freshness_score(NOW.isoformat(), now=NOW) == pytest.approx(1.0, abs=0.01)

    def test_half_life(self) -> None:
        old = (NOW - timedelta(days=365)).isoformat()
        assert freshness_score(old, half_life_days=365, now=NOW) == pytest.approx(0.5, abs=0.01)

    def test_unparseable_timestamp_neutral(self) -> None:
        assert freshness_score("not-a-date") == 0.5


class TestSourceQuality:
    def test_pdf_above_email(self) -> None:
        assert source_quality_score("pdf") > source_quality_score("email")

    def test_unknown_type_gets_default(self) -> None:
        assert source_quality_score("carrier_pigeon") == 0.70


class TestConsistency:
    def test_both_retrievers_beats_one(self) -> None:
        both = consistency_score(1, 1, 0.5, top_k=50)
        one = consistency_score(1, None, 0.5, top_k=50)
        assert both > one

    def test_rerank_contributes(self) -> None:
        high = consistency_score(1, 1, 1.0, top_k=50)
        low = consistency_score(1, 1, 0.0, top_k=50)
        assert high > low


class TestScoreChunk:
    def test_components_recorded(self) -> None:
        scored = score_chunk(make_scored(bm25_rank=1, vector_rank=2, rerank_score=0.8), now=NOW)
        assert set(scored.trust_components) == {"freshness", "source_quality", "consistency"}
        assert 0.0 <= scored.trust <= 1.0

    def test_fresh_agreeing_chunk_trusted(self) -> None:
        scored = score_chunk(
            make_scored(bm25_rank=1, vector_rank=1, rerank_score=0.9, source_type="pdf"),
            now=NOW,
        )
        assert scored.trust > 0.7

    def test_stale_single_retriever_chunk_less_trusted(self) -> None:
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
    def test_no_chunks_blocks(self) -> None:
        decision = evaluate_evidence([])
        assert not decision.allowed
        assert decision.reason == "no_chunks_retrieved"

    def test_low_trust_blocks(self) -> None:
        chunks = [make_scored(trust=0.1), make_scored(trust=0.2)]
        decision = evaluate_evidence(chunks, chunk_trust_floor=0.35)
        assert not decision.allowed
        assert decision.reason == "too_few_supporting_chunks"

    def test_confidence_threshold_blocks(self) -> None:
        chunks = [make_scored(trust=0.40)]
        decision = evaluate_evidence(chunks, confidence_threshold=0.60, chunk_trust_floor=0.35)
        assert not decision.allowed
        assert decision.reason == "confidence_below_threshold"

    def test_strong_evidence_allows(self) -> None:
        chunks = [make_scored(trust=0.8), make_scored(trust=0.7)]
        decision = evaluate_evidence(chunks)
        assert decision.allowed
        assert decision.confidence == pytest.approx(0.75)

    def test_answer_confidence_empty(self) -> None:
        assert answer_confidence([]) == 0.0


class TestExtractiveGenerator:
    def test_answer_sentences_verbatim_from_evidence(self) -> None:
        gen = ExtractiveGenerator()
        chunk = make_scored(
            "The rebate pays 900 dollars per kilowatt. Applications close in June.",
            trust=0.8,
        )
        answer, citations = gen.generate("How much does the rebate pay per kilowatt?", [chunk])
        assert "900 dollars" in answer
        for cite in citations:
            assert cite.quote in chunk.chunk.text  # zero-hallucination property

    def test_every_claim_cited(self) -> None:
        gen = ExtractiveGenerator()
        chunk = make_scored("Solar exports are capped at five kilowatts.", trust=0.9)
        answer, citations = gen.generate("What is the solar export cap?", [chunk])
        assert answer
        assert len(citations) >= 1
        assert citations[0].page == 1
        assert citations[0].source == "doc.txt"

    def test_irrelevant_evidence_yields_empty(self) -> None:
        gen = ExtractiveGenerator()
        chunk = make_scored("Bananas are yellow fruit.", trust=0.9)
        answer, citations = gen.generate("What is the tax rate on diesel?", [chunk])
        assert answer == ""
        assert citations == []


class TestPipelineEndToEnd:
    def test_answer_carries_citations_with_pages(self, pipeline) -> None:
        answer = pipeline.query("What is the maximum rebate per household?")
        assert answer.answered
        assert answer.citations
        for cite in answer.citations:
            assert cite.doc_id
            assert cite.page >= 1
            assert cite.quote

    def test_unanswerable_query_falls_back(self, pipeline) -> None:
        answer = pipeline.query("What is the average rainfall in the Amazon basin?")
        assert not answer.answered
        assert answer.text == INSUFFICIENT_EVIDENCE_MESSAGE
        assert answer.citations == []

    def test_empty_corpus_falls_back(self, empty_pipeline) -> None:
        answer = empty_pipeline.query("Anything at all?")
        assert not answer.answered
        assert answer.text == INSUFFICIENT_EVIDENCE_MESSAGE

    def test_confidence_reported(self, pipeline) -> None:
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
    def test_tier_boundaries(self, score, expected) -> None:
        from app.scoring.trust import trust_tier

        assert trust_tier(score) == expected


class TestWeightedAnswerConfidence:
    def test_empty_returns_zero(self) -> None:
        from app.scoring.trust import weighted_answer_confidence

        assert weighted_answer_confidence([]) == pytest.approx(0.0)

    def test_single_chunk(self) -> None:
        from app.scoring.trust import weighted_answer_confidence

        chunk = make_scored(trust=0.8, rerank_score=1.0)
        result = weighted_answer_confidence([chunk])
        assert 0.0 <= result <= 1.0

    def test_higher_rerank_weighs_more(self) -> None:
        from app.scoring.trust import weighted_answer_confidence

        low_rerank = make_scored(trust=0.5, rerank_score=0.1)
        high_rerank = make_scored(trust=0.9, rerank_score=0.9)
        result = weighted_answer_confidence([low_rerank, high_rerank])
        assert result > 0.5


class TestMinTrustGate:
    def test_empty_returns_empty(self) -> None:
        from app.scoring.trust import min_trust_gate

        assert min_trust_gate([]) == []

    def test_all_above_threshold_pass(self) -> None:
        from app.scoring.trust import min_trust_gate

        chunks = [make_scored(trust=0.8), make_scored(trust=0.9)]
        assert len(min_trust_gate(chunks, threshold=0.5)) == 2

    def test_below_threshold_filtered(self) -> None:
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
    def test_parametrized_gate(self, trusts, threshold, expected_count) -> None:
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


class TestExtractiveGeneratorEdgeCases:
    def test_empty_chunks_returns_empty_answer(self) -> None:
        """ExtractiveGenerator with no chunks returns empty answer and no citations."""
        from app.generation.extractive import ExtractiveGenerator

        gen = ExtractiveGenerator()
        answer, citations = gen.generate("What is the rebate?", [])
        assert answer == ""
        assert citations == []

    def test_empty_query_returns_empty(self) -> None:
        """ExtractiveGenerator with empty query returns empty."""
        from app.generation.extractive import ExtractiveGenerator

        gen = ExtractiveGenerator()
        answer, citations = gen.generate("", [make_scored(trust=0.8)])
        assert answer == ""
        assert citations == []

    def test_max_sentences_respected(self) -> None:
        """ExtractiveGenerator respects max_sentences limit."""
        from app.generation.extractive import ExtractiveGenerator
        from app.schemas import Chunk, ScoredChunk

        def make_chunk_with_text(text: str) -> ScoredChunk:
            chunk = Chunk(
                chunk_id="c1",
                doc_id="d1",
                source="test.txt",
                source_type="text",
                page=1,
                seq=0,
                text=text,
                created_at="2026-01-01T00:00:00+00:00",
            )
            return ScoredChunk(chunk=chunk, bm25_rank=1, vector_rank=1, rerank_score=0.9, trust=0.9)

        gen = ExtractiveGenerator(max_sentences=2)
        scored = make_chunk_with_text(
            "Solar rebates help households. Battery incentives boost storage. "
            "Grid rules govern export limits. Smart meters enable billing."
        )
        answer, citations = gen.generate("solar rebates battery incentives grid rules", [scored])
        sentence_count = len([s for s in answer.split(". ") if s.strip()])
        assert sentence_count <= 2

    def test_citations_match_answer_sentences(self) -> None:
        """Each citation claim corresponds to a sentence in the answer."""
        from app.generation.extractive import ExtractiveGenerator
        from app.schemas import Chunk, ScoredChunk

        chunk = Chunk(
            chunk_id="c1",
            doc_id="d1",
            source="pol.txt",
            source_type="text",
            page=1,
            seq=0,
            text="The rebate is nine hundred dollars per kilowatt.",
            created_at="2026-01-01T00:00:00+00:00",
        )
        scored = ScoredChunk(chunk=chunk, bm25_rank=1, vector_rank=1, rerank_score=0.9, trust=0.9)
        gen = ExtractiveGenerator()
        answer, citations = gen.generate("What is the rebate per kilowatt?", [scored])
        if citations:
            for citation in citations:
                assert citation.claim in answer or answer in citation.claim


class TestGateDecisionEdgeCases:
    def test_all_chunks_below_floor_blocks_generation(self) -> None:
        """All chunks below trust floor → gate blocks generation."""
        from app.generation.fallback import evaluate_evidence

        low_chunks = [make_scored(trust=0.2), make_scored(trust=0.1)]
        decision = evaluate_evidence(low_chunks, chunk_trust_floor=0.35)
        assert decision.allowed is False
        assert decision.reason == "too_few_supporting_chunks"

    def test_high_confidence_passes_gate(self) -> None:
        """Chunks with high trust clear the gate."""
        from app.generation.fallback import evaluate_evidence

        high_chunks = [make_scored(trust=0.9), make_scored(trust=0.85)]
        decision = evaluate_evidence(
            high_chunks, confidence_threshold=0.45, chunk_trust_floor=0.35
        )
        assert decision.allowed is True
        assert decision.reason == "evidence_sufficient"

    @pytest.mark.parametrize("n_chunks", [0, 1, 5])
    def test_empty_or_n_chunks(self, n_chunks: int) -> None:
        """Gate handles 0, 1, and several chunks without error."""
        from app.generation.fallback import evaluate_evidence

        chunks = [make_scored(trust=0.8) for _ in range(n_chunks)]
        decision = evaluate_evidence(chunks)
        assert isinstance(decision.allowed, bool)
        assert 0.0 <= decision.confidence <= 1.0
