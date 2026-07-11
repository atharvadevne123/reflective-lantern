"""Generate synthetic sensor data and seed the database or exercise the API."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def generate_readings(
    n: int = 500,
    sensor_id: str = "turbine-01",
    anomaly_rate: float = 0.03,
    seed: int = 42,
    start: datetime | None = None,
) -> list[dict]:
    """Generate synthetic multivariate readings with injected anomalies.

    Normal readings follow daily sinusoidal patterns with Gaussian noise;
    anomalies are large spikes in vibration and temperature.
    """
    rng = random.Random(seed)
    start = start or datetime(2026, 1, 1)
    readings = []
    for i in range(n):
        ts = start + timedelta(minutes=10 * i)
        day_phase = 2 * math.pi * (ts.hour * 60 + ts.minute) / 1440
        is_anomaly = rng.random() < anomaly_rate

        temperature = 60 + 10 * math.sin(day_phase) + rng.gauss(0, 1.5)
        vibration = 2.0 + 0.5 * math.sin(day_phase * 2) + rng.gauss(0, 0.2)
        rpm = 3000 + 200 * math.sin(day_phase) + rng.gauss(0, 30)
        power_kw = 450 + 80 * math.sin(day_phase) + rng.gauss(0, 12)

        if is_anomaly:
            vibration += rng.uniform(5, 15)
            temperature += rng.uniform(20, 40)

        readings.append(
            {
                "sensor_id": sensor_id,
                "timestamp": ts.isoformat(),
                "values": {
                    "temperature": round(temperature, 3),
                    "vibration": round(vibration, 3),
                    "rpm": round(rpm, 2),
                    "power_kw": round(power_kw, 2),
                },
                "_injected_anomaly": is_anomaly,
            }
        )
    injected = sum(1 for r in readings if r["_injected_anomaly"])
    logger.info("Generated %d readings (%d injected anomalies)", n, injected)
    return readings


def post_to_api(readings: list[dict], base_url: str) -> None:
    """Send readings to the /detect endpoint in batches."""
    import urllib.request

    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in readings]
    batch_size = 100
    detected = 0
    for i in range(0, len(clean), batch_size):
        payload = json.dumps({"readings": clean[i : i + batch_size], "horizon": 5}).encode()
        req = urllib.request.Request(
            f"{base_url}/api/v1/detect",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.load(resp)
                detected += result.get("anomaly_count", 0)
        except Exception as e:
            logger.error("Batch %d failed: %s", i // batch_size, e)
    logger.info("API detected %d anomalies total", detected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Temporal-Pulse with synthetic data")
    parser.add_argument("--n", type=int, default=500, help="Number of readings")
    parser.add_argument("--sensor-id", default="turbine-01")
    parser.add_argument("--anomaly-rate", type=float, default=0.03)
    parser.add_argument("--api-url", default=None, help="POST to this API instead of stdout")
    parser.add_argument("--output", default=None, help="Write JSON to this file")
    args = parser.parse_args()

    readings = generate_readings(args.n, args.sensor_id, args.anomaly_rate)

    if args.api_url:
        post_to_api(readings, args.api_url.rstrip("/"))
    elif args.output:
        with open(args.output, "w") as f:
            json.dump(readings, f, indent=2)
        logger.info("Wrote %s", args.output)
    else:
        print(json.dumps(readings[:3], indent=2))
        logger.info("(showing first 3 of %d; use --output or --api-url)", len(readings))


if __name__ == "__main__":
    main()
