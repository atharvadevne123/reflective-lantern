"""Tests for veritas-rag app/observability.py."""

from __future__ import annotations

import time

import pytest

from app.observability import Trace, TraceRecorder, TraceStore


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
