"""Thread-safe metrics counters and timing for Sport-Cast telemetry."""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_counters: dict[str, int] = {}
_timings: dict[str, list[float]] = {}


def increment(name: str, amount: int = 1) -> None:
    """Increment a named counter by the given amount.

    Args:
        name: Counter name.
        amount: Value to add (default 1).
    """
    with _lock:
        _counters[name] = _counters.get(name, 0) + amount


def record_timing(name: str, elapsed_ms: float) -> None:
    """Record a latency sample under the given name.

    Args:
        name: Metric name.
        elapsed_ms: Elapsed time in milliseconds.
    """
    with _lock:
        if name not in _timings:
            _timings[name] = []
        _timings[name].append(elapsed_ms)
        if len(_timings[name]) > 1000:
            _timings[name] = _timings[name][-1000:]


def snapshot() -> dict:
    """Return a point-in-time snapshot of all counters and timing percentiles.

    Returns:
        Dict with 'counters' and 'timings' sub-dicts.
    """
    import numpy as np

    with _lock:
        counters = dict(_counters)
        timings: dict[str, dict] = {}
        for name, samples in _timings.items():
            arr = np.array(samples)
            timings[name] = {
                "count": len(samples),
                "p50_ms": round(float(np.percentile(arr, 50)), 2),
                "p95_ms": round(float(np.percentile(arr, 95)), 2),
                "p99_ms": round(float(np.percentile(arr, 99)), 2),
            }
    return {"counters": counters, "timings": timings}


def reset() -> None:
    """Clear all counters and timing samples (testing / maintenance use)."""
    with _lock:
        _counters.clear()
        _timings.clear()
