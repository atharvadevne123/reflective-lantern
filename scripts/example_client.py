"""Example client for the Watt-Guard energy forecasting API."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime

import httpx

BASE_URL = "http://localhost:8000"
logger = logging.getLogger(__name__)


def predict(
    building_id: str,
    hour: int,
    day_of_week: int,
    month: int,
    temperature_c: float,
    humidity_pct: float,
    occupancy: int,
    hvac_state: int,
) -> dict:
    """Send a single prediction request to the Watt-Guard API.

    Args:
        building_id: Unique building identifier.
        hour: Hour of day (0-23).
        day_of_week: Day of week (0=Mon, 6=Sun).
        month: Month (1-12).
        temperature_c: Outside temperature in Celsius.
        humidity_pct: Relative humidity percentage.
        occupancy: Number of occupants.
        hvac_state: HVAC state (0=off, 1=on).

    Returns:
        Parsed JSON response dict.

    Raises:
        SystemExit: On HTTP or connection error.
    """
    payload = {
        "building_id": building_id,
        "timestamp": datetime.utcnow().isoformat(),
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "temperature_c": temperature_c,
        "humidity_pct": humidity_pct,
        "occupancy": occupancy,
        "hvac_state": hvac_state,
        "consumption_kwh": 0.0,
    }
    try:
        resp = httpx.post(f"{BASE_URL}/api/v1/predict", json=payload, timeout=10)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error("Request failed: %s", exc)
        sys.exit(1)
    return resp.json()


def main() -> None:
    """Run a sample prediction and print the result."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = predict(
        "bldg-001", hour=8, day_of_week=1, month=7, temperature_c=24.5, humidity_pct=55.0, occupancy=120, hvac_state=1
    )
    logger.info("Prediction result:\n%s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
