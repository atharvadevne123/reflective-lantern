"""Feature drift detection for fleet telemetry signals."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev


@dataclass
class DriftResult:
    feature: str
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float
    shift_z: float
    drift_detected: bool


def ks_statistic(reference: list[float], current: list[float]) -> float:
    """Approximate Kolmogorov-Smirnov statistic between two samples."""
    if not reference or not current:
        return 0.0
    all_vals = sorted(set(reference + current))
    ref_sorted = sorted(reference)
    cur_sorted = sorted(current)

    def ecdf(sample: list[float], x: float) -> float:
        return sum(1 for v in sample if v <= x) / len(sample)

    return max(abs(ecdf(ref_sorted, v) - ecdf(cur_sorted, v)) for v in all_vals)


def z_score_shift(reference: list[float], current: list[float]) -> float:
    """Return z-score of the shift in means (current mean vs reference distribution)."""
    if len(reference) < 2:
        return 0.0
    mu = mean(reference)
    sigma = stdev(reference)
    if sigma == 0:
        return 0.0
    cur_mean = mean(current) if current else mu
    return (cur_mean - mu) / sigma


def detect_drift(
    feature: str,
    reference: list[float],
    current: list[float],
    z_threshold: float = 2.0,
) -> DriftResult:
    """Detect mean-shift drift for a single feature."""
    ref_mean = mean(reference) if reference else 0.0
    cur_mean = mean(current) if current else 0.0
    ref_std = stdev(reference) if len(reference) > 1 else 0.0
    cur_std = stdev(current) if len(current) > 1 else 0.0
    shift = z_score_shift(reference, current)
    return DriftResult(
        feature=feature,
        reference_mean=ref_mean,
        current_mean=cur_mean,
        reference_std=ref_std,
        current_std=cur_std,
        shift_z=shift,
        drift_detected=abs(shift) > z_threshold,
    )


def batch_drift(
    features: dict[str, tuple[list[float], list[float]]],
    z_threshold: float = 2.0,
) -> list[DriftResult]:
    """Detect drift for multiple features at once."""
    return [detect_drift(f, ref, cur, z_threshold) for f, (ref, cur) in features.items()]
