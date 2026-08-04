"""Detect demand spikes using rolling Z-score and IQR methods."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def detect_spike(
    values: list[float],
    z_threshold: float = 2.5,
    iqr_multiplier: float = 1.5,
) -> dict[str, object]:
    """Return spike indices and statistics for a time-series window.

    Combines Z-score and IQR methods — a value is flagged as a spike
    if it exceeds either threshold.

    Args:
        values: Time-ordered energy consumption readings (kWh).
        z_threshold: Number of standard deviations above mean to flag.
        iqr_multiplier: IQR multiplier for the upper Tukey fence.

    Returns:
        Dict with 'spike_indices', 'spike_count', 'z_threshold',
        'iqr_upper', 'mean', and 'std'.  Returns minimal dict with
        'method'='insufficient_data' when fewer than 4 values provided.
    """
    arr = np.array(values, dtype=float)
    if len(arr) < 4:
        return {"spike_indices": [], "spike_count": 0, "method": "insufficient_data"}

    q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    iqr = q3 - q1
    iqr_upper = q3 + iqr_multiplier * iqr

    mean, std = float(arr.mean()), max(float(arr.std()), 1e-6)
    z_scores = np.abs((arr - mean) / std)

    spike_z = set(np.where(z_scores > z_threshold)[0].tolist())
    spike_iqr = set(np.where(arr > iqr_upper)[0].tolist())
    combined = sorted(spike_z | spike_iqr)

    if combined:
        logger.debug("detect_spike: %d spike(s) in %d readings", len(combined), len(values))

    return {
        "spike_indices": combined,
        "spike_count": len(combined),
        "z_threshold": z_threshold,
        "iqr_upper": round(iqr_upper, 4),
        "mean": round(mean, 4),
        "std": round(std, 4),
    }


def spike_severity(
    value: float,
    mean: float,
    std: float,
    z_threshold: float = 2.5,
) -> str:
    """Classify a single reading's spike severity.

    Args:
        value: The reading to classify.
        mean: Population mean.
        std: Population standard deviation (must be positive).
        z_threshold: Z-score base threshold.

    Returns:
        'critical' (>3× threshold), 'high' (>2× threshold),
        'moderate' (above threshold), or 'normal'.
    """
    if std < 1e-9:
        return "normal"
    z = abs(value - mean) / std
    if z > z_threshold * 3:
        return "critical"
    if z > z_threshold * 2:
        return "high"
    if z > z_threshold:
        return "moderate"
    return "normal"


def rolling_spike_count(
    values: list[float],
    window: int = 24,
    z_threshold: float = 2.5,
) -> list[int]:
    """Count spikes within a rolling window across the series.

    Args:
        values: Full time series of energy readings.
        window: Number of readings per rolling window.
        z_threshold: Z-score threshold passed to :func:`detect_spike`.

    Returns:
        List of spike counts, one per position in *values*
        (earlier positions with insufficient history return 0).
    """
    counts: list[int] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start : i + 1]
        result = detect_spike(window_vals, z_threshold=z_threshold)
        counts.append(int(result.get("spike_count", 0)))
    return counts


__all__ = [
    "detect_spike",
    "rolling_spike_count",
    "spike_severity",
]
