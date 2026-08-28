#!/usr/bin/env python3
"""Micro-benchmark suite for core app modules.

Usage::

    python scripts/benchmark.py [--runs N] [--module MODULE]
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable


def _timeit(fn: Callable, runs: int) -> list[float]:
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def _report(name: str, times: list[float]) -> None:
    mean_ms = statistics.mean(times) * 1000
    p95_ms = sorted(times)[int(len(times) * 0.95)] * 1000
    print(f"{name:<40} mean={mean_ms:8.3f}ms  p95={p95_ms:8.3f}ms  n={len(times)}")


def bench_retry(runs: int) -> None:
    from app.retry import retry

    @retry(max_attempts=1)
    def noop():
        return 42

    times = _timeit(noop, runs)
    _report("retry (no-op, 1 attempt)", times)


def bench_token_bucket(runs: int) -> None:
    from app.token_bucket import TokenBucket

    bucket = TokenBucket(capacity=10_000, refill_rate=10_000)
    times = _timeit(lambda: bucket.consume(1), runs)
    _report("token_bucket.consume(1)", times)


def bench_metrics_counter(runs: int) -> None:
    from app.metrics_collector import Counter

    c = Counter("bench")
    times = _timeit(lambda: c.inc(), runs)
    _report("Counter.inc()", times)


def bench_event_bus(runs: int) -> None:
    from app.event_bus import EventBus

    bus = EventBus()
    bus.subscribe("ev", lambda **kw: None)
    times = _timeit(lambda: bus.publish("ev", x=1), runs)
    _report("EventBus.publish (1 handler)", times)


def bench_haversine(runs: int) -> None:
    from app.geo_utils import Coordinate, haversine

    a = Coordinate(lat=51.5074, lon=-0.1278)
    b = Coordinate(lat=48.8566, lon=2.3522)
    times = _timeit(lambda: haversine(a, b), runs)
    _report("haversine (London → Paris)", times)


def bench_compression(runs: int) -> None:
    from app.compression import zlib_compress, zlib_decompress

    data = b"x" * 4096
    compressed = zlib_compress(data)
    times = _timeit(lambda: zlib_decompress(compressed), runs)
    _report("zlib_decompress (4 KB)", times)


BENCHMARKS = {
    "retry": bench_retry,
    "token_bucket": bench_token_bucket,
    "metrics": bench_metrics_counter,
    "event_bus": bench_event_bus,
    "haversine": bench_haversine,
    "compression": bench_compression,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run module benchmarks")
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--module", choices=list(BENCHMARKS), default=None)
    args = parser.parse_args()

    targets = {args.module: BENCHMARKS[args.module]} if args.module else BENCHMARKS
    print(f"Running {args.runs} iterations each\n")
    for fn in targets.values():
        fn(args.runs)


if __name__ == "__main__":
    main()
