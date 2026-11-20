"""Sensor noise filtering for telemetry signals."""

from __future__ import annotations

from collections import deque


def simple_moving_average(values: list[float], window: int = 5) -> list[float]:
    """Return simple moving average with the given window size."""
    if not values or window < 1:
        return list(values)
    result: list[float] = []
    buf: deque[float] = deque(maxlen=window)
    for v in values:
        buf.append(v)
        result.append(sum(buf) / len(buf))
    return result


def exponential_moving_average(values: list[float], alpha: float = 0.3) -> list[float]:
    """Return EMA with smoothing factor alpha (0 < alpha <= 1)."""
    if not values:
        return []
    alpha = max(0.0, min(1.0, alpha))
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result


def median_filter(values: list[float], window: int = 3) -> list[float]:
    """Apply median filter with given window; pads edges by replicating boundary values."""
    if not values or window < 1:
        return list(values)
    half = window // 2
    padded = [values[0]] * half + values + [values[-1]] * half
    result: list[float] = []
    for i in range(len(values)):
        segment = sorted(padded[i : i + window])
        result.append(segment[len(segment) // 2])
    return result


def remove_spikes(values: list[float], z_threshold: float = 3.0) -> list[float]:
    """Replace values more than z_threshold std devs from the mean with the mean."""
    if len(values) < 2:
        return list(values)
    mu = sum(values) / len(values)
    variance = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    sigma = variance**0.5
    if sigma == 0:
        return list(values)
    return [v if abs(v - mu) / sigma <= z_threshold else mu for v in values]


def clamp_values(values: list[float], lo: float, hi: float) -> list[float]:
    """Clamp each value to [lo, hi]."""
    return [max(lo, min(hi, v)) for v in values]
