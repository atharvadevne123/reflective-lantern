"""Model monitoring: KS-test drift detection and prediction logging."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import ks_2samp

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

DRIFT_THRESHOLD = 0.05  # p-value below this => drift detected


def compute_drift(
    reference: list[float],
    current: list[float],
    feature_name: str = "unknown",
) -> dict:
    """KS-test based drift detection between reference and current distributions."""
    if len(reference) < 10 or len(current) < 5:
        logger.warning("insufficient samples for drift test on %s", feature_name)
        return {
            "feature": feature_name,
            "ks_statistic": 0.0,
            "p_value": 1.0,
            "drift_detected": False,
            "note": "insufficient_samples",
        }

    stat, p_value = ks_2samp(reference, current)
    drift = bool(p_value < DRIFT_THRESHOLD)

    result = {
        "feature": feature_name,
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 4),
        "drift_detected": drift,
        "reference_n": len(reference),
        "current_n": len(current),
    }
    if drift:
        logger.warning("drift detected on %s: ks=%.4f p=%.4f", feature_name, stat, p_value)
    return result


def compute_all_feature_drifts(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_features: list[str] | None = None,
) -> list[dict]:
    """Compute drift across all numeric features."""

    if numeric_features is None:
        numeric_features = list(
            set(reference_df.select_dtypes(include=[np.number]).columns)
            & set(current_df.select_dtypes(include=[np.number]).columns)
        )

    results = []
    for feat in sorted(numeric_features):
        ref_vals = reference_df[feat].dropna().tolist()
        cur_vals = current_df[feat].dropna().tolist()
        results.append(compute_drift(ref_vals, cur_vals, feature_name=feat))

    return results


def compute_prediction_error_metrics(
    actual: list[float],
    predicted: list[float],
) -> dict:
    """RMSE, MAE and MAPE for a batch of predictions."""
    if not actual or not predicted or len(actual) != len(predicted):
        return {"rmse": None, "mae": None, "mape": None}

    a = np.array(actual, dtype=float)
    p = np.array(predicted, dtype=float)
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    mae = float(np.mean(np.abs(a - p)))
    mask = a != 0
    mape = float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100) if mask.any() else None

    return {"rmse": round(rmse, 3), "mae": round(mae, 3), "mape": round(mape, 3) if mape else None}


class PredictionTracker:
    """In-memory ring buffer for recent predictions — used for drift window."""

    def __init__(self, max_size: int = 1000) -> None:
        self._max = max_size
        self._loads: list[float] = []
        self._latencies: list[float] = []

    def record(self, load_mw: float, latency_ms: float) -> None:
        self._loads.append(float(load_mw))
        self._latencies.append(float(latency_ms))
        if len(self._loads) > self._max:
            self._loads = self._loads[-self._max:]
            self._latencies = self._latencies[-self._max:]

    @property
    def loads(self) -> list[float]:
        return list(self._loads)

    @property
    def latencies(self) -> list[float]:
        return list(self._latencies)

    @property
    def count(self) -> int:
        return len(self._loads)


_tracker = PredictionTracker()


def get_tracker() -> PredictionTracker:
    return _tracker


def compute_psi(
    reference: list[float],
    current: list[float],
    bins: int = 10,
) -> dict:
    """Population Stability Index (PSI) for distribution shift detection.

    Args:
        reference: Reference distribution values.
        current: Current distribution values to compare.
        bins: Number of histogram bins.

    Returns:
        dict with psi, severity ('stable'/'slight'/'significant'), and bin details.
    """
    import numpy as np

    if len(reference) < 10 or len(current) < 5:
        return {"psi": None, "severity": "unknown", "note": "insufficient_samples"}

    ref = np.array(reference, dtype=float)
    cur = np.array(current, dtype=float)
    global_min = min(ref.min(), cur.min())
    global_max = max(ref.max(), cur.max())

    if global_max == global_min:
        return {"psi": 0.0, "severity": "stable"}

    bin_edges = np.linspace(global_min, global_max, bins + 1)
    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    severity = "stable" if psi < 0.1 else "slight" if psi < 0.2 else "significant"

    return {"psi": round(psi, 4), "severity": severity, "bins": bins}
