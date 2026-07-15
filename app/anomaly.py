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


__all__ = ["anomaly_rate", "batch_compute_severity", "compute_severity", "iqr_flag", "zscore_flag"]
