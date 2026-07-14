"""Building energy efficiency scoring based on consumption patterns."""
from __future__ import annotations

import numpy as np


BUILDING_BENCHMARKS: dict[str, float] = {
    "residential": 4.5,
    "commercial": 12.0,
    "industrial": 30.0,
    "data_center": 60.0,
    "hospital": 25.0,
    "school": 8.0,
    "office": 10.0,
    "retail": 15.0,
}


def compute_efficiency_score(
    consumption_kwh: float,
    building_type: str,
    floor_area_sqm: float = 100.0,
) -> dict:
    """
    Score building energy efficiency (0-100, higher = more efficient).

    Uses consumption per square metre vs benchmark for the building type.
    """
    benchmark = BUILDING_BENCHMARKS.get(building_type.lower(), 10.0)
    consumption_per_sqm = consumption_kwh / max(floor_area_sqm, 1.0)
    ratio = consumption_per_sqm / max(benchmark / 100.0, 1e-6)
    score = float(np.clip(100.0 - (ratio - 1.0) * 50.0, 0.0, 100.0))
    grade = "A" if score >= 80 else "B" if score >= 60 else "C" if score >= 40 else "D"
    return {
        "score": round(score, 1),
        "grade": grade,
        "consumption_per_sqm": round(consumption_per_sqm, 4),
        "benchmark_per_sqm": round(benchmark / 100.0, 4),
        "building_type": building_type,
    }
