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


def spike_ratio(values: list[float], z_threshold: float = 2.5) -> float:
    """Return the fraction of values classified as spikes.

    Args:
        values: Time-series readings.
        z_threshold: Z-score threshold for spike detection.

    Returns:
        Ratio in [0, 1]; 0.0 when *values* is empty or has zero variance.
    """
    if len(values) < 2:
        return 0.0
    result = detect_spike(values, z_threshold=z_threshold)
    spike_indices = result.get("spike_indices", [])
    return len(spike_indices) / len(values)


def consecutive_spike_run(values: list[float], z_threshold: float = 2.5) -> int:
    """Return the length of the longest unbroken run of spike positions.

    A spike run is a contiguous sequence of consecutive indices all flagged
    as spikes by :func:`detect_spike`.

    Args:
        values: Time-series readings.
        z_threshold: Z-score threshold forwarded to :func:`detect_spike`.

    Returns:
        Length of the longest run; 0 if no spikes.
    """
    if not values:
        return 0
    result = detect_spike(values, z_threshold=z_threshold)
    spike_set = set(result.get("spike_indices", []))
    max_run = current_run = 0
    for i in range(len(values)):
        if i in spike_set:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def normalise_demand(values: list[float]) -> list[float]:
    """Return *values* min-max normalised to [0, 1].

    Args:
        values: Raw demand readings.

    Returns:
        Normalised list; all zeros when min == max.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


__all__ = [
    "detect_spike",
    "rolling_spike_count",
    "spike_severity",
    "spike_ratio",
    "consecutive_spike_run",
    "normalise_demand",
]


def spike_windows(
    values: list[float],
    threshold: float,
    window: int = 3,
) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for contiguous spike windows.

    Args:
        values: Input demand series.
        threshold: Spike threshold value.
        window: Minimum length of a spike window.

    Returns:
        List of (start, end) tuples (inclusive) for runs of spikes >= window long.
    """
    in_spike = [v > threshold for v in values]
    windows: list[tuple[int, int]] = []
    i = 0
    while i < len(in_spike):
        if in_spike[i]:
            j = i
            while j < len(in_spike) and in_spike[j]:
                j += 1
            if j - i >= window:
                windows.append((i, j - 1))
            i = j
        else:
            i += 1
    return windows


def demand_percentile(values: list[float], percentile: float = 95.0) -> float:
    """Return the *percentile*-th value of *values* using linear interpolation.

    Args:
        values: Demand readings.
        percentile: Target percentile in (0, 100].

    Returns:
        Interpolated percentile value.

    Raises:
        ValueError: If *values* is empty or *percentile* is out of range.
    """
    if not values:
        raise ValueError("values must not be empty")
    if not (0 < percentile <= 100):
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    idx = (percentile / 100.0) * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def smooth_demand(values: list[float], alpha: float = 0.3) -> list[float]:
    """Apply exponential smoothing (EMA) to a demand series.

    Args:
        values: Input demand readings.
        alpha: Smoothing factor in (0, 1]; higher = less smoothing.

    Returns:
        EMA-smoothed series the same length as *values*.

    Raises:
        ValueError: If *alpha* is out of range or *values* is empty.
    """
    if not values:
        raise ValueError("values must not be empty")
    if not (0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    result = [values[0]]
    for v in values[1:]:
        result.append(round(alpha * v + (1 - alpha) * result[-1], 6))
    return result
