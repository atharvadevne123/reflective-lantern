"""Stage 10 — full-pipeline tracing.

Every query produces a trace recording what each stage saw and decided:
retrieval paths (cache hit? BM25 ranking? vector ranking?), fusion and
rerank scores, per-chunk trust breakdowns, the gate decision, and the
claim→chunk attribution of the final answer. When the system answers
wrongly — or declines to answer — the trace states exactly why.
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Trace:
    """One request's end-to-end record."""

    request_id: str
    query: str
    started_at: float
    stages: list[dict[str, Any]] = field(default_factory=list)
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    outcome: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the trace to a plain dictionary suitable for JSON output."""
        return {
            "request_id": self.request_id,
            "query": self.query,
            "stages": self.stages,
            "stage_timings_ms": self.stage_timings_ms,
            "outcome": self.outcome,
        }


class TraceRecorder:
    """Builds one trace across pipeline stages."""

    def __init__(self, query: str) -> None:
        """Initialise a new trace for the given *query* string."""
        self.trace = Trace(request_id=str(uuid.uuid4()), query=query, started_at=time.monotonic())
        self._stage_started: float | None = None
        self._stage_name: str | None = None

    def start_stage(self, name: str) -> None:
        """Mark the start of pipeline stage *name*."""
        self._stage_name = name
        self._stage_started = time.monotonic()

    def end_stage(self, name: str, **data: Any) -> None:
        """Record the end of stage *name* together with arbitrary diagnostic *data*."""
        elapsed = 0.0
        if self._stage_started is not None and self._stage_name == name:
            elapsed = (time.monotonic() - self._stage_started) * 1000.0
        self.trace.stages.append({"stage": name, **data})
        self.trace.stage_timings_ms[name] = round(elapsed, 3)
        self._stage_started = None
        self._stage_name = None

    def finish(self, **outcome: Any) -> Trace:
        """Close the trace, attach the *outcome* dict, and return the completed Trace."""
        outcome["total_ms"] = round((time.monotonic() - self.trace.started_at) * 1000.0, 3)
        self.trace.outcome = outcome
        return self.trace


class TraceStore:
    """Ring buffer of recent traces, addressable by request id."""

    def __init__(self, max_size: int = 1000) -> None:
        """Initialise the store with a maximum capacity of *max_size* traces."""
        self.max_size = max_size
        self._traces: OrderedDict[str, Trace] = OrderedDict()

    def add(self, trace: Trace) -> None:
        """Add *trace* to the store, evicting the oldest entry when full."""
        self._traces[trace.request_id] = trace
        while len(self._traces) > self.max_size:
            self._traces.popitem(last=False)

    def get(self, request_id: str) -> Trace | None:
        """Return the trace for *request_id*, or None if not found."""
        return self._traces.get(request_id)

    def recent(self, limit: int = 20) -> list[Trace]:
        """Return the *limit* most recently added traces."""
        return list(self._traces.values())[-limit:]

    def __len__(self) -> int:
        """Return the number of traces currently held in the store."""
        return len(self._traces)


def average_latency_ms(store: TraceStore, stage: str | None = None) -> float:
    """Compute mean latency in milliseconds across recent traces.

    Args:
        store: A TraceStore instance to aggregate.
        stage: If given, average the latency of that specific pipeline stage;
            otherwise average the total request latency.

    Returns:
        Mean latency in ms; 0.0 when no traces exist.
    """
    traces = store.recent(limit=store.max_size)
    if not traces:
        return 0.0
    if stage is not None:
        values = [t.stage_timings_ms.get(stage, 0.0) for t in traces if stage in t.stage_timings_ms]
    else:
        values = [t.outcome.get("total_ms", 0.0) for t in traces if t.outcome]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def trace_error_rate(store: TraceStore) -> float:
    """Return the fraction of recent traces that completed with an error.

    Args:
        store: A TraceStore instance.

    Returns:
        Error rate in [0.0, 1.0]; 0.0 when no completed traces exist.
    """
    traces = [t for t in store.recent(limit=store.max_size) if t.outcome]
    if not traces:
        return 0.0
    errors = sum(1 for t in traces if t.outcome.get("error"))
    return round(errors / len(traces), 4)
