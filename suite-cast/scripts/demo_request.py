"""Send a demo prediction request against a running Suite-Cast instance.

Usage:
    python scripts/demo_request.py [--host http://localhost:8000]
"""

from __future__ import annotations

import argparse
import json

import httpx

DEMO_PAYLOAD: dict[str, object] = {
    "lead_time": 14,
    "length_of_stay": 3,
    "guests_count": 2,
    "checkin_month": 7,
    "checkin_dayofweek": 4,
    "is_weekend": 1,
    "current_occ_rate": 0.75,
    "prev_year_occ_rate": 0.70,
    "room_rate": 189.0,
    "competitor_avg_rate": 175.0,
    "special_event": 0,
    "room_type": "deluxe",
    "booking_channel": "online",
}


def main() -> int:
    """POST the demo payload and pretty-print the response.

    Returns:
        Process exit code (0 on success, 1 on HTTP failure).
    """
    parser = argparse.ArgumentParser(description="Suite-Cast demo client")
    parser.add_argument("--host", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()

    url = f"{args.host}/api/v1/predict"
    print(f"POST {url}")
    print(json.dumps(DEMO_PAYLOAD, indent=2))

    resp = httpx.post(url, json=DEMO_PAYLOAD, timeout=30.0)
    print(f"\nHTTP {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))
    return 0 if resp.status_code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
