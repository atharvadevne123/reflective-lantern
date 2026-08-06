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
        self.trace = Trace(request_id=str(uuid.uuid4()), query=query, started_at=time.monotonic())
        self._stage_started: float | None = None
        self._stage_name: str | None = None

    def start_stage(self, name: str) -> None:
        self._stage_name = name
        self._stage_started = time.monotonic()

    def end_stage(self, name: str, **data: Any) -> None:
        elapsed = 0.0
        if self._stage_started is not None and self._stage_name == name:
            elapsed = (time.monotonic() - self._stage_started) * 1000.0
        self.trace.stages.append({"stage": name, **data})
        self.trace.stage_timings_ms[name] = round(elapsed, 3)
        self._stage_started = None
        self._stage_name = None

    def finish(self, **outcome: Any) -> Trace:
        outcome["total_ms"] = round((time.monotonic() - self.trace.started_at) * 1000.0, 3)
        self.trace.outcome = outcome
        return self.trace


class TraceStore:
    """Ring buffer of recent traces, addressable by request id."""

    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self._traces: OrderedDict[str, Trace] = OrderedDict()

    def add(self, trace: Trace) -> None:
        self._traces[trace.request_id] = trace
        while len(self._traces) > self.max_size:
            self._traces.popitem(last=False)

    def get(self, request_id: str) -> Trace | None:
        return self._traces.get(request_id)

    def recent(self, limit: int = 20) -> list[Trace]:
        return list(self._traces.values())[-limit:]

    def __len__(self) -> int:
        return len(self._traces)
