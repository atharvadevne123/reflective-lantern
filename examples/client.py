"""Minimal Python client for the Logistics-Flow API."""
from __future__ import annotations

import argparse
import json
import urllib.request

DEFAULT_BASE = "http://localhost:8000"


def predict(
    base_url: str,
    *,
    carrier: str,
    distance_km: float,
    weight_kg: float,
    route_type: str,
    hour_of_day: int,
    day_of_week: int,
) -> dict:
    """POST a shipment to /api/v1/predict and return the parsed response."""
    payload = json.dumps(
        {
            "carrier": carrier,
            "distance_km": distance_km,
            "weight_kg": weight_kg,
            "route_type": route_type,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
        }
    ).encode()

    req = urllib.request.Request(
        f"{base_url}/api/v1/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a delivery time.")
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument("--carrier", default="DHL")
    parser.add_argument("--distance-km", type=float, default=42.5)
    parser.add_argument("--weight-kg", type=float, default=3.2)
    parser.add_argument("--route-type", default="urban")
    parser.add_argument("--hour-of-day", type=int, default=14)
    parser.add_argument("--day-of-week", type=int, default=2)
    args = parser.parse_args()

    result = predict(
        args.base_url,
        carrier=args.carrier,
        distance_km=args.distance_km,
        weight_kg=args.weight_kg,
        route_type=args.route_type,
        hour_of_day=args.hour_of_day,
        day_of_week=args.day_of_week,
    )
    print(f"{result['predicted_minutes']:.1f} min "
          f"({result['predicted_hours']:.2f} h), "
          f"confidence {result['confidence']:.2%}")


if __name__ == "__main__":
    main()
