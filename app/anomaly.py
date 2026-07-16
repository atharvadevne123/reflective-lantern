"""Extended anomaly analysis: Z-score, IQR, and multi-metric severity."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

ZSCORE_THRESHOLD = 2.5
IQR_MULTIPLIER = 1.5
RATIO_LOW_THRESHOLD = 0.4
RATIO_HIGH_THRESHOLD = 2.5
MIN_REFERENCE_SIZE = 4


def zscore_flag(value: float, mean: float, std: float, threshold: float = 3.0) -> bool:
    """Return True if *value* is more than *threshold* standard deviations from *mean*.

    Args:
        value: The observation to test.
        mean: Distribution mean.
        std: Distribution standard deviation.
        threshold: Number of standard deviations to use as the boundary.

    Returns:
        True when the observation is flagged as anomalous.
    """
    if std < 1e-9:
        return False
    return abs(value - mean) / std > threshold


def iqr_flag(value: float, q1: float, q3: float, k: float = 1.5) -> bool:
    """Return True if *value* is outside the Tukey fence defined by *q1*, *q3*, and *k*.

    Args:
        value: The observation to test.
        q1: First quartile of the reference distribution.
        q3: Third quartile of the reference distribution.
        k: IQR multiplier (default 1.5 = standard Tukey fence).

    Returns:
        True when the observation is flagged as anomalous.
    """
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return value < lower or value > upper


def compute_severity(
    value: float,
    reference: list[float],
    z_threshold: float = 3.0,
    iqr_k: float = 1.5,
) -> dict[str, object]:
    """Run both Z-score and IQR tests and combine into a severity label.

    Args:
        value: Consumption reading to evaluate.
        reference: Historical reference window.
        z_threshold: Z-score boundary for flagging.
        iqr_k: IQR fence multiplier.

    Returns:
        Dict with keys 'z_flag', 'iqr_flag', 'severity' ('none'|'warning'|'critical').
    """
    arr = np.array(reference, dtype=float)
    mean, std = float(arr.mean()), float(arr.std())
    q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))

    z = zscore_flag(value, mean, std, z_threshold)
    iq = iqr_flag(value, q1, q3, iqr_k)

    both = z and iq
    either = z or iq
    severity = "critical" if both else ("warning" if either else "none")

    logger.debug("Anomaly severity=%s z=%s iqr=%s value=%.2f mean=%.2f", severity, z, iq, value, mean)
    return {"z_flag": z, "iqr_flag": iq, "severity": severity}


def batch_compute_severity(
    values: list[float],
    reference: list[float],
    z_threshold: float = 3.0,
    iqr_k: float = 1.5,
) -> list[dict[str, object]]:
    """Run severity analysis on a batch of values against a shared reference.

    Args:
        values: List of consumption readings to evaluate.
        reference: Historical reference window (must have at least MIN_REFERENCE_SIZE elements).
        z_threshold: Z-score boundary for flagging.
        iqr_k: IQR fence multiplier.

    Returns:
        List of severity dicts (same order as *values*), each with 'z_flag', 'iqr_flag',
        'severity', and 'value' keys.
    """
    if len(reference) < MIN_REFERENCE_SIZE:
        logger.warning(
            "Reference window too small (%d < %d); returning 'none' for all values",
            len(reference),
            MIN_REFERENCE_SIZE,
        )
        return [{"z_flag": False, "iqr_flag": False, "severity": "none", "value": v} for v in values]
    results = []
    for v in values:
        result = compute_severity(v, reference, z_threshold, iqr_k)
        result["value"] = v
        results.append(result)
    return results


def anomaly_rate(severities: list[dict[str, object]]) -> float:
    """Return the fraction of readings flagged as anomalous (warning or critical).

    Args:
        severities: List of severity dicts as returned by batch_compute_severity.

    Returns:
        Float in [0, 1] representing the anomaly rate.
    """
    if not severities:
        return 0.0
    flagged = sum(1 for s in severities if s.get("severity") != "none")
    return round(flagged / len(severities), 4)


def compute_percentile_bounds(
    reference: list[float],
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> dict[str, float]:
    """Return lower and upper percentile bounds from a reference distribution.

    Args:
        reference: Historical reference window (at least 2 elements).
        lower_pct: Lower percentile (default 1st percentile).
        upper_pct: Upper percentile (default 99th percentile).

    Returns:
        Dict with 'lower', 'upper', 'median', and 'mean' keys.

    Raises:
        ValueError: If *reference* is empty or percentiles are out of [0, 100].
    """
    if not reference:
        raise ValueError("reference must be non-empty")
    if not (0 <= lower_pct < upper_pct <= 100):
        raise ValueError(f"Percentiles must satisfy 0 <= lower_pct < upper_pct <= 100, got {lower_pct}, {upper_pct}")
    arr = np.array(reference, dtype=float)
    return {
        "lower": round(float(np.percentile(arr, lower_pct)), 4),
        "upper": round(float(np.percentile(arr, upper_pct)), 4),
        "median": round(float(np.median(arr)), 4),
        "mean": round(float(arr.mean()), 4),
    }


def classify_consumption(
    value: float,
    low_threshold: float,
    high_threshold: float,
) -> str:
    """Classify a consumption reading as 'low', 'normal', or 'high'.

    Args:
        value: Consumption reading (kWh).
        low_threshold: Upper bound for the 'low' classification.
        high_threshold: Lower bound for the 'high' classification.

    Returns:
        'low' if value < low_threshold, 'high' if value >= high_threshold,
        else 'normal'.

    Raises:
        ValueError: If low_threshold >= high_threshold.
    """
    if low_threshold >= high_threshold:
        raise ValueError(
            f"low_threshold must be less than high_threshold, got {low_threshold} >= {high_threshold}"
        )
    if value < low_threshold:
        return "low"
    if value >= high_threshold:
        return "high"
    return "normal"


__all__ = [
    "anomaly_rate",
    "batch_compute_severity",
    "classify_consumption",
    "compute_percentile_bounds",
    "compute_severity",
    "iqr_flag",
    "top_anomalies",
    "zscore_flag",
]


def top_anomalies(
    severities: list[dict[str, object]],
    n: int = 10,
    severity_order: list[str] | None = None,
) -> list[dict[str, object]]:
    """Return the *n* most severe entries from a severity result list.

    Args:
        severities: List of dicts with at least a 'severity' key (from batch_compute_severity).
        n: Maximum number of results to return.
        severity_order: Ordered list from most to least severe; defaults to
            ['critical', 'warning', 'none'].

    Returns:
        Sorted slice of the input list, most severe first, up to length *n*.
    """
    order = severity_order or ["critical", "warning", "none"]
    rank = {label: i for i, label in enumerate(order)}
    default_rank = len(order)

    def _key(entry: dict[str, object]) -> int:
        return rank.get(str(entry.get("severity", "")), default_rank)

    return sorted(severities, key=_key)[:n]
