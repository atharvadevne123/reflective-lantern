"""Latency benchmarking script for Ticket-Oracle.

Fires N sequential /predict requests against a running server and
prints p50/p95/p99 latency statistics.

Usage:
    python scripts/benchmark.py --url http://localhost:8000 --n 100
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from typing import Any

SAMPLE_PAYLOAD: dict[str, Any] = {
    "ticket_id": "BENCH-001",
    "description": "Production database is unreachable causing widespread service outage for all users",
    "department": "IT",
    "incident_type": "outage",
    "channel": "email",
    "customer_tier": "Gold",
    "product_area": "database",
    "open_tickets_count": 12,
    "agent_load": 0.85,
    "hour_of_day": 10,
    "day_of_week": 1,
    "ticket_age_minutes": 5.0,
    "same_user_last_7d": 0,
}


def percentile(data: list[float], p: float) -> float:
    data_sorted = sorted(data)
    idx = int(len(data_sorted) * p / 100)
    return data_sorted[min(idx, len(data_sorted) - 1)]


def run_benchmark(base_url: str, n: int) -> None:
    url = f"{base_url}/api/v1/predict"
    body = json.dumps(SAMPLE_PAYLOAD).encode()
    latencies: list[float] = []

    for i in range(n):
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
        except Exception as exc:
            print(f"Request {i+1} failed: {exc}")
            continue
        latencies.append((time.perf_counter() - t0) * 1000)

    if not latencies:
        print("No successful requests.")
        return

    print(f"\nBenchmark results — {len(latencies)}/{n} successful requests")
    print(f"  mean   : {statistics.mean(latencies):.2f} ms")
    print(f"  p50    : {percentile(latencies, 50):.2f} ms")
    print(f"  p95    : {percentile(latencies, 95):.2f} ms")
    print(f"  p99    : {percentile(latencies, 99):.2f} ms")
    print(f"  min    : {min(latencies):.2f} ms")
    print(f"  max    : {max(latencies):.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ticket-Oracle latency benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="Base server URL")
    parser.add_argument("--n", type=int, default=50, help="Number of requests")
    args = parser.parse_args()
    run_benchmark(args.url, args.n)


if __name__ == "__main__":
    main()
