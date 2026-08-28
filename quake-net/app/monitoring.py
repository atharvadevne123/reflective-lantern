"""Drift detection and prediction logging for seismic model monitoring."""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

DRIFT_WINDOW = int(os.getenv("DRIFT_WINDOW", "200"))
REFERENCE_PATH = Path(os.getenv("REFERENCE_PATH", "reference_dist.json"))
DRIFT_ALPHA = float(os.getenv("DRIFT_ALPHA", "0.05"))


class PredictionStore:
    """Thread-safe in-memory store for recent predictions (for drift checks)."""

    def __init__(self, max_size: int = DRIFT_WINDOW) -> None:
        self._lock = Lock()
        self._store: dict[str, deque[float]] = {}
        self._max_size = max_size

    def record(self, features: dict[str, Any], prediction: float) -> None:
        """Append one prediction's numeric features and output to the windows."""
        with self._lock:
            for key, val in features.items():
                if not isinstance(val, (int, float)):
                    continue
                if key not in self._store:
                    self._store[key] = deque(maxlen=self._max_size)
                self._store[key].append(float(val))
            if "prediction" not in self._store:
                self._store["prediction"] = deque(maxlen=self._max_size)
            self._store["prediction"].append(prediction)

    def get_feature_window(self, feature: str) -> list[float]:
        """Return a snapshot copy of the retained values for ``feature``."""
        with self._lock:
            return list(self._store.get(feature, []))

    def all_features(self) -> list[str]:
        """Return every feature name currently being tracked."""
        with self._lock:
            return list(self._store.keys())

    def sample_count(self, feature: str) -> int:
        """Return how many samples are retained for ``feature``."""
        with self._lock:
            return len(self._store.get(feature, []))


_store = PredictionStore()


def get_store() -> PredictionStore:
    """Return the process-wide prediction store used for drift checks."""
    return _store


def compute_drift(reference: list[float], current: list[float]) -> dict[str, Any]:
    """Two-sample Kolmogorov-Smirnov test between reference and current values.

    Returns:
        ``ks_statistic``, ``p_value`` and a ``drift_detected`` flag set when the
        p-value falls below ``DRIFT_ALPHA``. Samples too small to test return
        ``drift_detected: False`` with an ``error`` key rather than raising —
        an unseeded feature is not evidence of drift.
    """
    if len(reference) < 2 or len(current) < 2:
        return {
            "ks_statistic": None,
            "p_value": None,
            "drift_detected": False,
            "error": "insufficient samples",
        }
    stat, p = ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < DRIFT_ALPHA),
    }


def load_reference_distribution() -> dict[str, list[float]]:
    """Load the stored drift baseline, or an empty mapping when unseeded."""
    if not REFERENCE_PATH.exists():
        return {}
    return json.loads(REFERENCE_PATH.read_text())


def save_reference_distribution(dist: dict[str, list[float]]) -> None:
    """Persist a per-feature drift baseline to ``REFERENCE_PATH``."""
    REFERENCE_PATH.write_text(json.dumps(dist))


def check_all_drift(db_session=None) -> list[dict[str, Any]]:
    """Run KS-test on each tracked feature against reference distribution."""
    from app.database import DriftLog

    reference = load_reference_distribution()
    results = []

    for feature in _store.all_features():
        current = _store.get_feature_window(feature)
        ref_data = reference.get(feature, [])

        if not ref_data:
            logger.debug("No reference for feature %s — skipping drift check", feature)
            continue

        drift = compute_drift(ref_data, current)
        drift["feature_name"] = feature
        drift["sample_size"] = len(current)
        drift["checked_at"] = datetime.utcnow().isoformat()
        results.append(drift)

        if db_session is not None and drift.get("ks_statistic") is not None:
            try:
                record = DriftLog(
                    feature_name=feature,
                    ks_statistic=drift["ks_statistic"],
                    p_value=drift["p_value"],
                    drift_detected=drift["drift_detected"],
                    sample_size=drift["sample_size"],
                )
                db_session.add(record)
                db_session.commit()
            except Exception:
                db_session.rollback()
                logger.exception("Failed to write drift log to DB")

        if drift.get("drift_detected"):
            logger.warning(
                "DRIFT DETECTED on feature '%s' — KS=%.4f p=%.4f",
                feature,
                drift.get("ks_statistic", 0),
                drift.get("p_value", 1),
            )

    return results


def compute_psi(reference: list[float], current: list[float], bins: int = 10) -> float:
    """Population Stability Index — complementary to KS test."""
    if len(reference) < 10 or len(current) < 10:
        return 0.0
    edges = np.histogram_bin_edges(reference, bins=bins)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / (ref_counts.sum() + 1e-9)
    cur_pct = cur_counts / (cur_counts.sum() + 1e-9)

    # PSI = sum((actual - expected) * ln(actual / expected))
    psi = float(np.sum((cur_pct - ref_pct) * np.log((cur_pct + 1e-9) / (ref_pct + 1e-9))))
    return round(psi, 4)
