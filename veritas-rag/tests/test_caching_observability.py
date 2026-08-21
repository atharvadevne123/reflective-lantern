"""Stages 9-10 tests: retrieval cache and tracing."""

from __future__ import annotations

import time

from app.caching import RetrievalCache, cache_key, normalize_query
from app.observability import TraceRecorder, TraceStore


class TestCache:
    def test_hit_after_put(self):
        cache = RetrievalCache()
        cache.put("What is X?", ["result"])
        assert cache.get("What is X?") == ["result"]
        assert cache.stats()["hits"] == 1

    def test_normalized_variants_share_entry(self):
        cache = RetrievalCache()
        cache.put("What is the Rebate?", "value")
        assert cache.get("what is the rebate") == "value"
        assert normalize_query("What IS   the rebate??") == "what is the rebate"

    def test_ttl_expiry(self):
        cache = RetrievalCache(ttl_seconds=0.01)
        cache.put("q", "v")
        time.sleep(0.02)
        assert cache.get("q") is None
        assert cache.stats()["misses"] == 1

    def test_lru_eviction(self):
        cache = RetrievalCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # refresh a
        cache.put("c", 3)  # evicts b
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3

    def test_invalidate_all(self):
        cache = RetrievalCache()
        cache.put("a", 1)
        cache.invalidate_all()
        assert cache.get("a") is None

    def test_key_stability(self):
        assert cache_key("Query!") == cache_key("query")


class TestTraceRecorder:
    def test_stages_and_timings_recorded(self):
        rec = TraceRecorder("test query")
        rec.start_stage("bm25")
        rec.end_stage("bm25", top=["c1"])
        trace = rec.finish(answered=True)
        assert trace.stages[0]["stage"] == "bm25"
        assert "bm25" in trace.stage_timings_ms
        assert trace.outcome["answered"] is True
        assert "total_ms" in trace.outcome

    def test_request_ids_unique(self):
        ids = {TraceRecorder("q").trace.request_id for _ in range(50)}
        assert len(ids) == 50


class TestTraceStore:
    def test_ring_buffer_eviction(self):
        store = TraceStore(max_size=2)
        traces = [TraceRecorder(f"q{i}").finish() for i in range(3)]
        for t in traces:
            store.add(t)
        assert store.get(traces[0].request_id) is None
        assert store.get(traces[2].request_id) is not None
        assert len(store) == 2

    def test_recent_order(self):
        store = TraceStore()
        traces = [TraceRecorder(f"q{i}").finish() for i in range(3)]
        for t in traces:
            store.add(t)
        recent = store.recent(2)
        assert [t.query for t in recent] == ["q1", "q2"]


class TestPipelineTracing:
    def test_query_produces_full_trace(self, pipeline):
        answer = pipeline.query("What is the maximum rebate per household?")
        trace = pipeline.traces.get(answer.request_id)
        assert trace is not None
        stage_names = {s["stage"] for s in trace.stages}
        assert {
            "cache",
            "bm25",
            "vector",
            "fusion",
            "rerank",
            "trust",
            "evidence_gate",
        } <= stage_names

    def test_trace_explains_fallback(self, pipeline):
        answer = pipeline.query("What is the melting point of osmium?")
        trace = pipeline.traces.get(answer.request_id)
        assert trace.outcome["answered"] is False
        assert trace.outcome["reason"]

    def test_cache_hit_recorded_in_trace(self, pipeline):
        q = "How much is the rooftop solar rebate per kilowatt?"
        pipeline.query(q)
        answer2 = pipeline.query(q)
        trace = pipeline.traces.get(answer2.request_id)
        cache_stage = next(s for s in trace.stages if s["stage"] == "cache")
        assert cache_stage["hit"] is True

    def test_attribution_present_for_answers(self, pipeline):
        answer = pipeline.query("What is the maximum rebate per household?")
        trace = pipeline.traces.get(answer.request_id)
        gen_stage = next(s for s in trace.stages if s["stage"] == "generation")
        assert gen_stage["attribution"]
        assert all("chunk_id" in a for a in gen_stage["attribution"])

    def test_cached_second_query_faster_path(self, pipeline):
        q = "What are the grid export limits for single phase connections?"
        pipeline.query(q)
        answer = pipeline.query(q)
        trace = pipeline.traces.get(answer.request_id)
        # Cached path skips retrieval stages entirely
        stage_names = [s["stage"] for s in trace.stages]
        assert "bm25" not in stage_names


class TestCacheFillRatio:
    def test_empty_cache_is_zero(self) -> None:
        from app.caching import RetrievalCache, cache_fill_ratio

        c = RetrievalCache(max_size=10)
        assert cache_fill_ratio(c) == 0.0

    def test_full_cache_is_one(self) -> None:
        from app.caching import RetrievalCache, cache_fill_ratio

        c = RetrievalCache(max_size=2)
        c.put("q1", "v1")
        c.put("q2", "v2")
        assert cache_fill_ratio(c) == 1.0

    def test_partial_fill(self) -> None:
        from app.caching import RetrievalCache, cache_fill_ratio

        c = RetrievalCache(max_size=4)
        c.put("q1", "v1")
        assert 0.0 < cache_fill_ratio(c) <= 0.5


class TestTopCachedQueries:
    def test_empty_cache_returns_empty(self) -> None:
        from app.caching import RetrievalCache, top_cached_queries

        c = RetrievalCache()
        assert top_cached_queries(c) == []

    def test_returns_at_most_n(self) -> None:
        from app.caching import RetrievalCache, top_cached_queries

        c = RetrievalCache()
        for i in range(10):
            c.put(f"query_{i}", f"val_{i}")
        result = top_cached_queries(c, n=3)
        assert len(result) == 3

    def test_returns_list_of_strings(self) -> None:
        from app.caching import RetrievalCache, top_cached_queries

        c = RetrievalCache()
        c.put("hello world", "v")
        result = top_cached_queries(c, n=5)
        assert all(isinstance(k, str) for k in result)


class TestAverageLatencyMs:
    def test_empty_store_returns_zero(self) -> None:
        from app.observability import TraceStore, average_latency_ms

        assert average_latency_ms(TraceStore()) == 0.0

    def test_missing_stage_returns_zero(self) -> None:
        from app.observability import TraceStore, average_latency_ms

        s = TraceStore()
        assert average_latency_ms(s, stage="nonexistent") == 0.0


class TestTraceErrorRate:
    def test_empty_store_returns_zero(self) -> None:
        from app.observability import TraceStore, trace_error_rate

        assert trace_error_rate(TraceStore()) == 0.0

    def test_returns_float(self) -> None:
        from app.observability import TraceStore, trace_error_rate

        result = trace_error_rate(TraceStore())
        assert isinstance(result, float)
