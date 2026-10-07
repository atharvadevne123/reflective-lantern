"""Z-score and IQR anomaly detection for Sports-Oracle feature monitoring."""

from __future__ import annotations

import numpy as np


def zscore_anomalies(values: list[float], threshold: float = 3.0) -> list[int]:
    """Return indices of values more than `threshold` standard deviations from the mean."""
    if len(values) < 3:
        return []
    arr = np.array(values)
    mean, std = arr.mean(), arr.std()
    if std < 1e-10:
        return []
    zscores = np.abs((arr - mean) / std)
    return [int(i) for i in np.where(zscores > threshold)[0]]


def iqr_anomalies(values: list[float], multiplier: float = 1.5) -> list[int]:
    """Return indices of values outside the IQR fence."""
    if len(values) < 4:
        return []
    arr = np.array(values)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return [int(i) for i in np.where((arr < lower) | (arr > upper))[0]]


def detect_anomalies(
    values: list[float],
    zscore_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
) -> dict:
    """Run both Z-score and IQR anomaly detection and return combined report."""
    z_idxs = zscore_anomalies(values, threshold=zscore_threshold)
    iqr_idxs = iqr_anomalies(values, multiplier=iqr_multiplier)
    all_idxs = sorted(set(z_idxs) | set(iqr_idxs))
    return {
        "n_samples": len(values),
        "n_anomalies": len(all_idxs),
        "anomaly_indices": all_idxs,
        "anomaly_rate": round(len(all_idxs) / max(len(values), 1), 4),
        "zscore_indices": z_idxs,
        "iqr_indices": iqr_idxs,
    }
