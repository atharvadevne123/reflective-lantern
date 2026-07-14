"""Peak demand prediction: identify highest-consumption hours in a forecast window."""
from __future__ import annotations

import numpy as np


def find_peak_hours(
    hourly_forecasts: list[float],
    top_n: int = 3,
) -> dict:
    """Return the top-N peak consumption hours from a 24h forecast window."""
    if not hourly_forecasts:
        return {"peak_hours": [], "peak_values": [], "avg_kwh": 0.0}
    arr = np.array(hourly_forecasts, dtype=float)
    indices = np.argsort(arr)[-top_n:][::-1].tolist()
    return {
        "peak_hours": [int(i) for i in indices],
        "peak_values": [round(float(arr[i]), 4) for i in indices],
        "avg_kwh": round(float(arr.mean()), 4),
        "total_kwh": round(float(arr.sum()), 4),
        "peak_to_avg_ratio": round(float(arr[indices[0]] / max(arr.mean(), 1e-6)), 4),
    }


def estimate_peak_shaving_savings(
    hourly_forecasts: list[float],
    shave_pct: float = 0.15,
    cost_per_kwh: float = 0.12,
) -> dict:
    """Estimate cost savings from peak shaving the top consumption hours."""
    arr = np.array(hourly_forecasts, dtype=float)
    peak_threshold = np.percentile(arr, (1 - shave_pct) * 100)
    shaved = np.where(arr > peak_threshold, peak_threshold, arr)
    savings_kwh = float((arr - shaved).sum())
    return {
        "savings_kwh": round(savings_kwh, 4),
        "savings_cost": round(savings_kwh * cost_per_kwh, 4),
        "shave_pct": shave_pct,
        "peak_threshold_kwh": round(float(peak_threshold), 4),
    }
