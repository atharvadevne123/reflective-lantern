"""Pipeline-level edge case tests."""

from __future__ import annotations

import pytest

from app.pipeline import RagPipeline


class TestPipelineEdgeCases:
    def test_empty_index_returns_insufficient_evidence(self, empty_pipeline: RagPipeline) -> None:
        """Querying an empty pipeline returns an insufficient-evidence answer."""
        answer = empty_pipeline.query("What is the rebate amount?")
        assert answer.answered is False
        assert "Insufficient" in answer.text

    def test_query_returns_request_id(self, empty_pipeline: RagPipeline) -> None:
        """Every query response includes a non-empty request_id."""
        answer = empty_pipeline.query("What is the rebate amount?")
        assert answer.request_id
        assert len(answer.request_id) > 0

    def test_trace_stored_after_query(self, empty_pipeline: RagPipeline) -> None:
        """After a query, the trace is retrievable by request_id."""
        answer = empty_pipeline.query("What is the rebate amount?")
        trace = empty_pipeline.traces.get(answer.request_id)
        assert trace is not None
        assert trace.query == "What is the rebate amount?"

    def test_cache_hit_on_repeated_query(self, pipeline: RagPipeline) -> None:
        """A repeated query hits the cache and returns the same answer."""
        q = "How much is the solar rebate per kilowatt?"
        first = pipeline.query(q)
        second = pipeline.query(q)
        assert first.text == second.text
        stats = pipeline.cache.stats()
        assert stats["hits"] >= 1

    def test_ingest_invalidates_cache(self, pipeline: RagPipeline) -> None:
        """Re-ingesting a document clears the cache."""
        q = "What is the solar rebate amount per kilowatt?"
        pipeline.query(q)
        pipeline.ingest(
            b"Updated solar rebate is now 1000 dollars per kilowatt.",
            "cache-buster.txt",
        )
        assert len(pipeline.cache) == 0

    def test_metrics_includes_expected_keys(self, pipeline: RagPipeline) -> None:
        """pipeline.metrics() returns all expected top-level keys."""
        m = pipeline.metrics()
        required = {"documents", "chunks", "bm25_index_size", "vector_index_size", "cache"}
        assert required.issubset(m.keys())

    def test_query_stats_increment(self, empty_pipeline: RagPipeline) -> None:
        """pipeline.stats['queries'] increments with each query call."""
        assert empty_pipeline.stats["queries"] == 0
        empty_pipeline.query("first query here")
        empty_pipeline.query("second query here")
        assert empty_pipeline.stats["queries"] == 2

    def test_remove_document_chunks_cleans_index(self, empty_pipeline: RagPipeline) -> None:
        """_remove_document_chunks evicts all chunks for the given doc_id."""
        report = empty_pipeline.ingest(b"Document to be removed content.", "remove-me.txt")
        doc_id = report["doc_id"]
        assert any(c.doc_id == doc_id for c in empty_pipeline.chunks.values())
        empty_pipeline._remove_document_chunks(doc_id)
        assert not any(c.doc_id == doc_id for c in empty_pipeline.chunks.values())

    def test_confidence_in_valid_range(self, pipeline: RagPipeline) -> None:
        """Answer confidence is always in [0, 1]."""
        answer = pipeline.query("What is the solar rebate per kilowatt?")
        assert 0.0 <= answer.confidence <= 1.0

    @pytest.mark.parametrize(
        "question",
        [
            "What is the maximum household solar rebate?",
            "How many days does application processing take?",
            "What department administers the program?",
        ],
    )
    def test_answered_questions_have_citations(
        self, pipeline: RagPipeline, question: str
    ) -> None:
        """Answered questions always include at least one citation."""
        answer = pipeline.query(question)
        if answer.answered:
            assert len(answer.citations) >= 1
            for citation in answer.citations:
                assert citation.source
                assert citation.chunk_id


class TestQueryBatch:
    def test_batch_returns_correct_count(self, pipeline) -> None:
        answers = pipeline.query_batch(["What is the solar rebate?", "What is the export limit?"])
        assert len(answers) == 2

    def test_batch_empty_list(self, pipeline) -> None:
        answers = pipeline.query_batch([])
        assert answers == []

    def test_batch_each_answer_has_request_id(self, pipeline) -> None:
        answers = pipeline.query_batch(["solar rebate?", "battery incentive?"])
        for a in answers:
            assert a.request_id

    def test_batch_increments_query_stats(self, pipeline) -> None:
        before = pipeline.stats["queries"]
        pipeline.query_batch(["q1", "q2", "q3"])
        assert pipeline.stats["queries"] == before + 3


class TestDocumentCount:
    def test_zero_on_empty_pipeline(self, empty_pipeline) -> None:
        assert empty_pipeline.document_count() == 0

    def test_count_increments_per_source(self, empty_pipeline) -> None:
        empty_pipeline.ingest(b"First document for count.", "count-a.txt")
        assert empty_pipeline.document_count() == 1
        empty_pipeline.ingest(b"Second document for count.", "count-b.txt")
        assert empty_pipeline.document_count() == 2

    def test_same_source_version_does_not_increment(self, empty_pipeline) -> None:
        empty_pipeline.ingest(b"Original version of the document.", "versioned.txt")
        empty_pipeline.ingest(b"Updated version of the document.", "versioned.txt")
        assert empty_pipeline.document_count() == 1

    def test_count_matches_sources_length(self, pipeline) -> None:
        assert pipeline.document_count() == len(pipeline.sources())
