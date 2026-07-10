"""Extended anomaly analysis: Z-score, IQR, and multi-metric severity."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


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
    """Return True if *value* falls outside the IQR fence.

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
