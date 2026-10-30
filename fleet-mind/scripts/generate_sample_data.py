"""Generate sample telemetry data for local testing and demo purposes."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path


def generate_sample_readings(
    n: int = 100,
    vehicle_ids: list[str] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Generate synthetic fleet telemetry readings.

    Args:
        n: Number of readings to generate.
        vehicle_ids: List of vehicle IDs to cycle through.
        seed: Random seed for reproducibility.

    Returns:
        List of telemetry dicts suitable for the /predict/batch endpoint.
    """
    random.seed(seed)
    if vehicle_ids is None:
        vehicle_ids = [f"VH-{i:03d}" for i in range(1, 11)]

    readings = []
    for i in range(n):
        vehicle_id = vehicle_ids[i % len(vehicle_ids)]
        is_anomalous = random.random() < 0.2

        reading = {
            "vehicle_id": vehicle_id,
            "speed_kmh": round(random.uniform(0, 130), 1),
            "engine_temp_c": round(random.uniform(95, 130) if is_anomalous else random.uniform(65, 95), 1),
            "oil_pressure_bar": round(random.uniform(0.5, 2.0) if is_anomalous else random.uniform(2.5, 5.5), 2),
            "fuel_consumption_lph": round(random.uniform(4, 12), 2),
            "battery_voltage_v": round(random.uniform(10.5, 12.0) if is_anomalous else random.uniform(12.2, 13.5), 2),
            "brake_pad_mm": round(random.uniform(0.5, 3.0) if is_anomalous else random.uniform(3.5, 12.0), 1),
            "tire_pressure_bar": round(random.uniform(1.2, 1.8) if is_anomalous else random.uniform(2.0, 2.8), 2),
            "rpm": round(random.uniform(700, 5000), 0),
            "coolant_temp_c": round(random.uniform(88, 115) if is_anomalous else random.uniform(68, 90), 1),
            "transmission_temp_c": round(random.uniform(110, 145) if is_anomalous else random.uniform(60, 90), 1),
        }
        readings.append(reading)

    return readings


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/fleet_sample_data.json")

    readings = generate_sample_readings(n=n)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"readings": readings}, fh, indent=2)

    print(f"Generated {n} readings -> {out_path}")


if __name__ == "__main__":
    main()
