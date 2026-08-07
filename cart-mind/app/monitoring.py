"""Drift detection, prediction logging, and request counters.

Two complementary drift signals are reported for every feature. The KS test
answers "is this shift statistically real?", while PSI answers "is it big enough
to care about?" — a distinction that matters in production, where enough traffic
makes almost any shift statistically significant.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from threading import Lock
from typing import Any

import numpy as np
from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import DriftLog, PredictionLog

logger = logging.getLogger(__name__)

# Rolling window for drift detection (per feature)
_REFERENCE_WINDOW: dict[str, deque] = {}
_REFERENCE_LOCK = Lock()
REFERENCE_WINDOW_SIZE = 500
DRIFT_P_THRESHOLD = 0.05


def compute_drift(reference: list[float], current: list[float]) -> dict[str, Any]:
    """Run a two-sample KS test between reference and current distributions.

    Args:
        reference: Baseline values from the rolling reference window.
        current: Recently observed values.

    Returns:
        Mapping with the KS statistic, p-value, and drift flag. Samples under 20
        observations return no verdict with ``reason: "insufficient_data"``,
        since a KS test on a handful of points is not worth acting on.
    """
    if len(reference) < 20 or len(current) < 20:
        return {
            "ks_statistic": None,
            "p_value": None,
            "drift_detected": False,
            "reason": "insufficient_data",
        }
    stat, p = ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < DRIFT_P_THRESHOLD),
    }


def update_reference_window(feature_name: str, values: list[float]) -> None:
    """Append observations to a feature's rolling reference window.

    The window is a bounded deque, so the baseline tracks recent behaviour rather
    than growing without limit — drift is measured against the recent past, not
    against everything ever seen.

    Args:
        feature_name: Feature whose window should be extended.
        values: Observations to append.
    """
    with _REFERENCE_LOCK:
        if feature_name not in _REFERENCE_WINDOW:
            _REFERENCE_WINDOW[feature_name] = deque(maxlen=REFERENCE_WINDOW_SIZE)
        _REFERENCE_WINDOW[feature_name].extend(values)


def get_reference_window(feature_name: str) -> list[float]:
    with _REFERENCE_LOCK:
        return list(_REFERENCE_WINDOW.get(feature_name, []))


def check_all_features(
    feature_values: dict[str, list[float]],
    db: Session,
) -> dict[str, dict[str, Any]]:
    """Check every supplied feature for drift and persist the results.

    Args:
        feature_values: Map of feature name to recently observed values.
        db: Session used to write one :class:`DriftLog` row per feature.

    Returns:
        Per-feature results carrying both the KS verdict and the PSI severity.
    """
    results: dict[str, dict[str, Any]] = {}
    for feat, current in feature_values.items():
        reference = get_reference_window(feat)
        result = compute_drift(reference, current)
        # PSI is reported alongside KS so a caller can weigh statistical
        # significance against effect size before deciding to retrain.
        psi_result = compute_psi(reference, current)
        result["psi"] = psi_result.get("psi")
        result["psi_severity"] = psi_result.get("severity")
        results[feat] = result
        if result.get("ks_statistic") is not None:
            db.add(
                DriftLog(
                    feature_name=feat,
                    ks_statistic=result["ks_statistic"],
                    p_value=result["p_value"],
                    drift_detected=int(result["drift_detected"]),
                    window_size=len(current),
                )
            )
    db.commit()
    drifted = [f for f, r in results.items() if r.get("drift_detected")]
    if drifted:
        logger.warning("Drift detected in features: %s", drifted)
    return results


def log_prediction(  # noqa: PLR0913
    db: Session,
    correlation_id: str,
    user_id: str,
    item_id: str | None,
    prediction_type: str,
    score: float,
    latency_ms: float,
    model_version: str = "1.0.0",
) -> None:
    """Persist one prediction record.

    Args:
        db: Database session.
        correlation_id: Request correlation ID, for tracing a row to a request.
        user_id: User the prediction was made for.
        item_id: Item scored, or ``None`` for batch and recommendation calls.
        prediction_type: One of ``intent``, ``recommend``, ``similar``, ``batch``.
        score: Model output being recorded.
        latency_ms: Server-side inference latency.
        model_version: Version string of the serving model.
    """
    db.add(
        PredictionLog(
            correlation_id=correlation_id,
            user_id=user_id,
            item_id=item_id,
            prediction_type=prediction_type,
            score=score,
            model_version=model_version,
            latency_ms=round(latency_ms, 2),
        )
    )
    db.commit()


# ---------------------------------------------------------------------------
# In-memory metrics counters (for /metrics endpoint)
# ---------------------------------------------------------------------------


class _Counters:
    """Thread-safe request counters and a rolling latency window.

    Latencies live in a bounded deque, so percentiles describe recent traffic and
    memory stays flat regardless of uptime.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self.total_requests = 0
        self.intent_requests = 0
        self.recommend_requests = 0
        self.similar_requests = 0
        self.drift_alerts = 0
        self.errors = 0
        self._latencies: deque[float] = deque(maxlen=1000)
        self._start = time.time()

    def record(self, route: str, latency_ms: float, error: bool = False) -> None:
        """Record one request against the counters.

        Args:
            route: Route label — ``intent``, ``recommend``, or ``similar``.
            latency_ms: Observed latency in milliseconds.
            error: Whether the request failed.
        """
        with self._lock:
            self.total_requests += 1
            if route == "intent":
                self.intent_requests += 1
            elif route == "recommend":
                self.recommend_requests += 1
            elif route == "similar":
                self.similar_requests += 1
            if error:
                self.errors += 1
            self._latencies.append(latency_ms)

    def snapshot(self) -> dict[str, Any]:
        """Return a point-in-time view of the counters.

        Returns:
            Counts, uptime, and p50/p95/p99 latency over the rolling window.
        """
        with self._lock:
            lats = list(self._latencies)
        return {
            "uptime_seconds": round(time.time() - self._start, 1),
            "total_requests": self.total_requests,
            "intent_requests": self.intent_requests,
            "recommend_requests": self.recommend_requests,
            "similar_requests": self.similar_requests,
            "drift_alerts": self.drift_alerts,
            "errors": self.errors,
            "p50_latency_ms": round(float(np.percentile(lats, 50)), 2) if lats else 0.0,
            "p95_latency_ms": round(float(np.percentile(lats, 95)), 2) if lats else 0.0,
            "p99_latency_ms": round(float(np.percentile(lats, 99)), 2) if lats else 0.0,
        }


COUNTERS = _Counters()


# ---------------------------------------------------------------------------
# Population Stability Index
# ---------------------------------------------------------------------------

# Conventional PSI reading: <0.10 stable, 0.10-0.25 moderate shift, >0.25 major shift.
PSI_MODERATE_THRESHOLD = 0.10
PSI_MAJOR_THRESHOLD = 0.25


def compute_psi(reference: list[float], current: list[float], bins: int = 10) -> dict[str, Any]:
    """Compute the Population Stability Index between two distributions.

    PSI complements the KS test rather than duplicating it. KS is sensitive to
    the single largest gap between two CDFs and returns a p-value that grows
    more significant purely with sample size — on a high-traffic endpoint it
    will eventually flag drift that is real but far too small to matter. PSI
    instead sums relative frequency shifts across quantile bins, giving a
    magnitude that is stable under sample size and reads directly as an effect
    size against long-established thresholds.

    Args:
        reference: Baseline distribution values.
        current: Recent distribution values to compare against the baseline.
        bins: Number of quantile bins to discretise into.

    Note:
        PSI is noisy on small samples — at n=200 across 10 bins there are only
        ~20 observations per bin, which can inflate the score past the
        ``moderate`` threshold on two samples drawn from the same distribution.
        Prefer a few hundred observations per side before acting on a reading.

    Returns:
        Mapping with the PSI value, a severity label, and a drift flag. When
        either sample is too small, returns ``psi: None`` with a reason.
    """
    if len(reference) < 20 or len(current) < 20:
        return {
            "psi": None,
            "severity": "unknown",
            "drift_detected": False,
            "reason": "insufficient_data",
        }

    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    # Quantile edges from the reference, so bins carry equal baseline mass.
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return {
            "psi": None,
            "severity": "unknown",
            "drift_detected": False,
            "reason": "degenerate_reference",
        }
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=edges)[0] / len(cur)

    # Floor empty bins: an unpopulated bin would otherwise send PSI to infinity.
    floor = 1e-4
    ref_pct = np.clip(ref_pct, floor, None)
    cur_pct = np.clip(cur_pct, floor, None)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    if psi >= PSI_MAJOR_THRESHOLD:
        severity = "major"
    elif psi >= PSI_MODERATE_THRESHOLD:
        severity = "moderate"
    else:
        severity = "stable"

    return {
        "psi": round(psi, 4),
        "severity": severity,
        "drift_detected": psi >= PSI_MODERATE_THRESHOLD,
        "n_bins": len(edges) - 1,
    }
