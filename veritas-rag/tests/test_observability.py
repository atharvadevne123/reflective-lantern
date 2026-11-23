"""Tests for veritas-rag app/observability.py."""

from __future__ import annotations

import time

from app.observability import Trace, TraceRecorder, TraceStore, average_latency_ms, trace_error_rate


class TestTrace:
    def test_create(self) -> None:
        trace = Trace(
            request_id="r1",
            query="test query",
            started_at=time.monotonic(),
        )
        assert trace.request_id == "r1"
        assert trace.query == "test query"

    def test_to_dict(self) -> None:
        trace = Trace(request_id="r1", query="q", started_at=time.monotonic())
        d = trace.to_dict()
        assert isinstance(d, dict)
        assert "request_id" in d
        assert "query" in d

    def test_stages_default_empty(self) -> None:
        trace = Trace(request_id="r1", query="q", started_at=time.monotonic())
        assert trace.stages == []


class TestTraceRecorder:
    def test_create(self) -> None:
        recorder = TraceRecorder("what is AI?")
        assert recorder.trace is not None
        assert recorder.trace.query == "what is AI?"

    def test_unique_request_ids(self) -> None:
        r1 = TraceRecorder("query1")
        r2 = TraceRecorder("query2")
        assert r1.trace.request_id != r2.trace.request_id

    def test_record_stage(self) -> None:
        recorder = TraceRecorder("query")
        recorder.record_stage("retrieval", {"hits": 3})
        assert len(recorder.trace.stages) == 1

    def test_finish_returns_trace(self) -> None:
        recorder = TraceRecorder("query")
        trace = recorder.finish(outcome={"answer": "42"})
        assert isinstance(trace, Trace)
        assert trace.outcome == {"answer": "42"}


class TestTraceStore:
    def test_empty_initially(self) -> None:
        store = TraceStore(max_traces=10)
        assert len(store.list()) == 0

    def test_add_trace(self) -> None:
        store = TraceStore(max_traces=10)
        recorder = TraceRecorder("q")
        trace = recorder.finish({})
        store.add(trace)
        assert len(store.list()) == 1

    def test_max_traces_limit(self) -> None:
        store = TraceStore(max_traces=3)
        for i in range(5):
            r = TraceRecorder(f"q{i}")
            store.add(r.finish({}))
        assert len(store.list()) <= 3

    def test_get_by_request_id(self) -> None:
        store = TraceStore()
        recorder = TraceRecorder("query")
        trace = recorder.finish({})
        store.add(trace)
        result = store.get(trace.request_id)
        assert result is not None
        assert result.request_id == trace.request_id


def _make_trace(query: str = "q", answered: bool = True, total_ms: float = 10.0) -> Trace:
    recorder = TraceRecorder(query)
    recorder.start_stage("retrieval")
    recorder.end_stage("retrieval", hits=3)
    return recorder.finish(answered=answered, total_ms=total_ms)


class TestTraceRecorderStages:
    def test_start_end_stage_records_timing(self) -> None:
        recorder = TraceRecorder("q")
        recorder.start_stage("bm25")
        recorder.end_stage("bm25", results=5)
        assert "bm25" in recorder.trace.stage_timings_ms

    def test_stage_timing_non_negative(self) -> None:
        recorder = TraceRecorder("q")
        recorder.start_stage("test")
        recorder.end_stage("test")
        assert recorder.trace.stage_timings_ms["test"] >= 0.0

    def test_multiple_stages_recorded(self) -> None:
        recorder = TraceRecorder("q")
        for stage in ["bm25", "vector", "rerank", "trust"]:
            recorder.start_stage(stage)
            recorder.end_stage(stage)
        assert len(recorder.trace.stages) == 4

    def test_finish_has_total_ms(self) -> None:
        recorder = TraceRecorder("q")
        trace = recorder.finish(answered=True)
        assert "total_ms" in trace.outcome
        assert trace.outcome["total_ms"] >= 0.0


class TestTraceStoreRecent:
    def test_recent_returns_latest(self) -> None:
        store = TraceStore(max_size=10)
        for i in range(5):
            store.add(_make_trace(query=f"q{i}"))
        recent = store.recent(limit=3)
        assert len(recent) == 3

    def test_recent_all_when_fewer_than_limit(self) -> None:
        store = TraceStore(max_size=10)
        store.add(_make_trace(query="only one"))
        assert len(store.recent(limit=50)) == 1

    def test_len_counts_stored(self) -> None:
        store = TraceStore(max_size=10)
        for i in range(4):
            store.add(_make_trace(query=f"q{i}"))
        assert len(store) == 4

    def test_missing_request_id_returns_none(self) -> None:
        store = TraceStore()
        assert store.get("nonexistent-id") is None


class TestAverageLatency:
    def test_empty_store_returns_zero(self) -> None:
        store = TraceStore()
        assert average_latency_ms(store) == 0.0

    def test_computes_mean_total(self) -> None:
        store = TraceStore(max_size=100)
        recorder = TraceRecorder("q")
        trace = recorder.finish(answered=True)
        store.add(trace)
        result = average_latency_ms(store)
        assert result >= 0.0

    def test_stage_latency_zero_when_absent(self) -> None:
        store = TraceStore(max_size=100)
        recorder = TraceRecorder("q")
        store.add(recorder.finish())
        result = average_latency_ms(store, stage="bm25")
        assert result == 0.0


class TestTraceErrorRate:
    def test_no_traces_returns_zero(self) -> None:
        store = TraceStore()
        assert trace_error_rate(store) == 0.0

    def test_no_errors_returns_zero(self) -> None:
        store = TraceStore(max_size=10)
        for _ in range(3):
            recorder = TraceRecorder("q")
            store.add(recorder.finish(answered=True))
        assert trace_error_rate(store) == 0.0

    def test_all_errors_returns_one(self) -> None:
        store = TraceStore(max_size=10)
        for _ in range(3):
            recorder = TraceRecorder("q")
            store.add(recorder.finish(error=True))
        assert trace_error_rate(store) == 1.0
