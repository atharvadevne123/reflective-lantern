"""Example client for the Traffic-Pulse API."""

from __future__ import annotations

import json
import sys

import httpx

BASE_URL = "http://localhost:8000"


def main() -> None:
    """Send a sample prediction request and print the response."""
    payload = {
        "route_id": "R42",
        "hour": 8,
        "day_of_week": 1,
        "vehicle_count": 1800,
        "avg_speed_kmh": 32.5,
        "road_type": "arterial",
        "incident_count": 1,
        "is_raining": 1,
    }
    try:
        resp = httpx.post(f"{BASE_URL}/api/v1/predict", json=payload, timeout=10)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(resp.json(), indent=2))


if __name__ == "__main__":
    main()
