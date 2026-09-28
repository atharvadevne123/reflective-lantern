"""Post-deploy smoke test: exercises every endpoint against a live server."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

SAMPLE = {
    "carrier": "DHL",
    "distance_km": 42.5,
    "weight_kg": 3.2,
    "route_type": "urban",
    "hour_of_day": 14,
    "day_of_week": 2,
}


def _get(url: str) -> tuple[int, dict]:
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, json.load(r)


def _post(url: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    base = parser.parse_args().base_url.rstrip("/")

    failures: list[str] = []

    status, health = _get(f"{base}/api/v1/health")
    if status != 200 or health.get("status") != "healthy":
        failures.append(f"health returned {status} {health}")

    status, pred = _post(f"{base}/api/v1/predict", SAMPLE)
    if status != 200 or pred.get("predicted_minutes", 0) <= 0:
        failures.append(f"predict returned {status} {pred}")

    status, bad = _post(f"{base}/api/v1/predict", {**SAMPLE, "carrier": "Nope"})
    if status != 422:
        failures.append(f"invalid carrier should be 422, got {status}")

    status, _ = _get(f"{base}/api/v1/metrics")
    if status != 200:
        failures.append(f"metrics returned {status}")

    status, _ = _get(f"{base}/api/v1/drift")
    if status != 200:
        failures.append(f"drift returned {status}")

    if failures:
        logger.error("SMOKE TEST FAILED")
        for f in failures:
            logger.error("  - %s", f)
        return 1

    logger.info(
        "SMOKE TEST PASSED — predicted %.1f min at %.2f%% confidence",
        pred["predicted_minutes"],
        pred["confidence"] * 100,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
