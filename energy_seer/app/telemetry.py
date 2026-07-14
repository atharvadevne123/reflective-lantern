"""Request and prediction telemetry counters."""
from __future__ import annotations

from collections import defaultdict

_counters: dict[str, int] = defaultdict(int)


def increment(name: str, amount: int = 1) -> None:
    _counters[name] += amount


def get(name: str) -> int:
    return _counters[name]


def snapshot() -> dict[str, int]:
    return dict(_counters)


def reset(name: str | None = None) -> None:
    if name:
        _counters[name] = 0
    else:
        _counters.clear()
