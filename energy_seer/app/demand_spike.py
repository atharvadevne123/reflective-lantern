"""Detect demand spikes using rolling Z-score and IQR methods."""
from __future__ import annotations

import numpy as np


def detect_spike(
    values: list[float],
    z_threshold: float = 2.5,
    iqr_multiplier: float = 1.5,
) -> dict:
    """Return spike indices and statistics for a time-series window."""
    arr = np.array(values, dtype=float)
    if len(arr) < 4:
        return {"spike_indices": [], "method": "insufficient_data"}

    q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    iqr = q3 - q1
    iqr_upper = q3 + iqr_multiplier * iqr

    mean, std = float(arr.mean()), max(float(arr.std()), 1e-6)
    z_scores = np.abs((arr - mean) / std)

    spike_z = set(np.where(z_scores > z_threshold)[0].tolist())
    spike_iqr = set(np.where(arr > iqr_upper)[0].tolist())
    combined = sorted(spike_z | spike_iqr)

    return {
        "spike_indices": combined,
        "spike_count": len(combined),
        "z_threshold": z_threshold,
        "iqr_upper": round(iqr_upper, 4),
        "mean": round(mean, 4),
        "std": round(std, 4),
    }
