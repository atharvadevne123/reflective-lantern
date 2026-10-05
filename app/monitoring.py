"""Prediction logging, KS-test drift detection, and monitoring utilities."""
from __future__ import annotations

import logging
import statistics
import time
from collections import deque
from datetime import datetime
from typing import Any

from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import AnomalyLog, DriftLog, Prediction, PredictionLog

logger = logging.getLogger(__name__)

# In-memory rolling buffer for reference window (populated at startup)
_REFERENCE_BUFFER: dict[str, deque] = {
    "distance_km": deque(maxlen=500),
    "weight_kg": deque(maxlen=500),
    "predicted_minutes": deque(maxlen=500),
}

# Single-feature reference window for energy-domain drift monitoring
_reference_window: list[float] = []

# Global anomaly flags buffer for no-arg rolling_anomaly_rate
_anomaly_flags_buffer: list[bool] = []


def reset_anomaly_flags_buffer() -> None:
    """Clear the in-memory anomaly flags buffer (test helper)."""
    global _anomaly_flags_buffer
    _anomaly_flags_buffer = []

# Global alert counts
_global_alert_counts: dict[str, int] = {"warning": 0, "error": 0, "critical": 0}


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------

class LatencyTimer:
    """Context manager that records elapsed wall-clock time in milliseconds."""

    def __enter__(self) -> LatencyTimer:
        self._start = time.perf_counter()
        self.ms: float = 0.0
        return self

    def __exit__(self, *_: object) -> None:
        self.ms = (time.perf_counter() - self._start) * 1000.0


# ---------------------------------------------------------------------------
# Reference window management
# ---------------------------------------------------------------------------

def set_reference_window(values: list[float]) -> None:
    """Replace the in-memory reference window (capped at 500 samples)."""
    global _reference_window
    _reference_window = list(values[-500:])


def reset_reference_window() -> None:
    """Clear the reference window."""
    global _reference_window
    _reference_window = []


def get_reference_window_size() -> int:
    """Return the number of samples currently in the reference window."""
    return len(_reference_window)


def is_reference_window_ready(min_samples: int = 10) -> bool:
    """Return True when the reference window has at least *min_samples* entries."""
    return len(_reference_window) >= min_samples


def reference_window_stats() -> dict[str, Any]:
    """Return descriptive statistics for the current reference window."""
    n = len(_reference_window)
    if n == 0:
        return {"size": 0, "mean": None, "min": None, "max": None, "std": None}
    mean = sum(_reference_window) / n
    mn = min(_reference_window)
    mx = max(_reference_window)
    std = statistics.pstdev(_reference_window)
    return {"size": n, "mean": mean, "min": mn, "max": mx, "std": std}


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def compute_drift(reference: list[float], current: list[float]) -> dict[str, Any]:
    """Run KS test between reference and current distributions."""
    if len(reference) < 10 or len(current) < 10:
        return {
            "ks_statistic": 0.0,
            "p_value": 1.0,
            "drift_detected": False,
            "reason": "insufficient_data",
        }
    stat, p = ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < 0.05),
    }


def compute_feature_drift_summary(
    features: dict[str, list[float]],
    reference: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Compute drift for each feature against the shared or provided reference window."""
    ref = reference if reference is not None else _reference_window
    results = []
    for feature_name, current_values in features.items():
        result = compute_drift(ref, current_values)
        result["feature"] = feature_name
        results.append(result)
    return results


def summarize_drift_history(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a list of drift-check results into a summary."""
    total = len(results)
    if total == 0:
        return {
            "total_checks": 0,
            "drift_count": 0,
            "drift_rate": 0.0,
            "mean_ks_statistic": None,
            "min_p_value": None,
        }
    drift_count = sum(1 for r in results if r.get("drift_detected"))
    ks_values = [r["ks_statistic"] for r in results if r.get("ks_statistic") is not None]
    p_values = [r["p_value"] for r in results if r.get("p_value") is not None]
    return {
        "total_checks": total,
        "drift_count": drift_count,
        "drift_rate": drift_count / total,
        "mean_ks_statistic": sum(ks_values) / len(ks_values) if ks_values else None,
        "min_p_value": min(p_values) if p_values else None,
    }


def drift_severity(p_value: float) -> str:
    """Classify drift severity from a KS-test p-value (lower = more severe)."""
    if p_value <= 0.001:
        return "critical"
    if p_value <= 0.01:
        return "high"
    if p_value < 0.05:
        return "medium"
    return "low"


def drift_trend(p_values: list[float]) -> str:
    """Detect whether drift is worsening, improving, or stable over a p-value series."""
    if len(p_values) < 3:
        return "stable"
    half = len(p_values) // 2
    mean_first = sum(p_values[:half]) / half
    mean_second = sum(p_values[half:]) / (len(p_values) - half)
    if mean_second < mean_first * 0.8:
        return "worsening"
    if mean_second > mean_first * 1.2:
        return "improving"
    return "stable"


# ---------------------------------------------------------------------------
# Anomaly detection helpers
# ---------------------------------------------------------------------------

def zscore_alert(values: list[float], threshold: float = 3.0) -> list[int]:
    """Return indices of values whose z-score exceeds *threshold*.

    Raises ValueError for a series shorter than 2 or a non-positive threshold.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if len(values) < 2:
        raise ValueError("need at least 2 values to compute z-scores")
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5
    if std < 1e-12:
        return []
    return [i for i, v in enumerate(values) if abs(v - mean) / std > threshold]


def rolling_anomaly_rate(
    flags: list[bool] | None = None,
    window: int = 3,
) -> list[float] | float:
    """Compute rolling anomaly rate.

    With *flags*: returns a list of rates over sliding windows of size *window*.
    Without *flags*: returns a single float from the global anomaly buffer.
    """
    if flags is None:
        if not _anomaly_flags_buffer:
            return 0.0
        tail = _anomaly_flags_buffer[-window:]
        return sum(tail) / len(tail)
    if window <= 0:
        raise ValueError("window must be positive")
    if not flags:
        return []
    n = len(flags)
    if n < window:
        return []
    return [
        sum(flags[i : i + window]) / window
        for i in range(n - window + 1)
    ]


# ---------------------------------------------------------------------------
# Alerting helpers
# ---------------------------------------------------------------------------

def alert_count_by_level(alerts: list[dict[str, Any]] | None = None) -> dict[str, int]:
    """Count alerts by their 'level' key.

    With no argument returns the global accumulated counts.
    With an empty list returns an empty dict.
    """
    if alerts is None:
        return dict(_global_alert_counts)
    counts: dict[str, int] = {}
    for alert in alerts:
        level = alert.get("level", "unknown")
        counts[level] = counts.get(level, 0) + 1
    return counts


def alert_suppression_window(
    last_alert_ts: float,
    current_ts: float,
    cooldown_seconds: float,
) -> bool:
    """Return True if the alert should be suppressed (within cooldown window)."""
    if cooldown_seconds <= 0:
        raise ValueError("cooldown_seconds must be positive")
    return (current_ts - last_alert_ts) < cooldown_seconds


def alert_rate(values: list[float], window: int = 3) -> float:
    """Return the mean of the last *window* values (rolling mean tail)."""
    if not values:
        return 0.0
    tail = values[-window:] if len(values) >= window else values
    return sum(tail) / len(tail)


# ---------------------------------------------------------------------------
# Degradation / SLO helpers
# ---------------------------------------------------------------------------

def degradation_severity(error_rate: float) -> str:
    """Classify service degradation from an observed error rate."""
    if error_rate <= 0.0:
        return "ok"
    if error_rate < 0.01:
        return "healthy"
    if error_rate < 0.05:
        return "low"
    if error_rate < 0.10:
        return "warning"
    if error_rate < 0.20:
        return "medium"
    if error_rate < 0.30:
        return "critical"
    return "high"


def error_budget_remaining(
    slo_or_total: float = 0.0,
    actual_or_errors: float = 0.0,
    period_minutes: float = 0.0,
    *,
    total_requests: float | None = None,
    error_count: float | None = None,
    slo_target: float | None = None,
) -> float:
    """Compute remaining error budget.

    Supports two calling conventions:
    - ``error_budget_remaining(slo_target, actual_avail, period_minutes=M)``
    - ``error_budget_remaining(total_requests=N, error_count=E, slo_target=T)``
    """
    if total_requests is not None and error_count is not None and slo_target is not None:
        allowed = (1.0 - slo_target) * total_requests
        return allowed - error_count
    # Positional: (slo_target, actual_availability, period_minutes)
    slo = slo_or_total
    actual = actual_or_errors
    budget_minutes = (1.0 - slo) * period_minutes
    consumed_minutes = (1.0 - actual) * period_minutes
    return budget_minutes - consumed_minutes


def p_value_to_confidence(p_value: float) -> float:
    """Convert a KS-test p-value to a confidence percentage (0-100)."""
    clamped = max(0.0, min(1.0, p_value))
    return (1.0 - clamped) * 100.0


# ---------------------------------------------------------------------------
# Prediction and anomaly logging (logistics domain)
# ---------------------------------------------------------------------------

def log_prediction(
    db: Session,
    *args: Any,
    carrier: str = "",
    distance_km: float = 0.0,
    weight_kg: float = 0.0,
    route_type: str = "",
    hour_of_day: int = 0,
    day_of_week: int = 0,
    predicted_minutes: float = 0.0,
    confidence: float = 0.0,
    model_version: str = "1.0.0",
) -> Any:
    """Persist a prediction record.

    Supports two calling conventions:
    - ``log_prediction(db, building_id, timestamp, predicted_kwh, latency_ms)`` (energy domain)
    - ``log_prediction(db, carrier=..., distance_km=..., ...)`` (logistics domain)
    """
    if args and isinstance(args[0], str):
        # Energy domain: (db, building_id, timestamp, predicted_kwh, latency_ms)
        building_id = args[0]
        ts = args[1] if len(args) > 1 else datetime.utcnow()
        predicted_kwh = float(args[2]) if len(args) > 2 else 0.0
        latency_ms = float(args[3]) if len(args) > 3 else None
        pred = PredictionLog(
            building_id=building_id,
            timestamp=ts,
            predicted_kwh=predicted_kwh,
            latency_ms=latency_ms,
        )
        db.add(pred)
        db.commit()
        db.refresh(pred)
        return pred

    # Logistics domain
    pred = Prediction(
        created_at=datetime.utcnow(),
        carrier=carrier,
        distance_km=distance_km,
        weight_kg=weight_kg,
        route_type=route_type,
        hour_of_day=hour_of_day,
        day_of_week=day_of_week,
        predicted_minutes=predicted_minutes,
        confidence=confidence,
        model_version=model_version,
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)

    _REFERENCE_BUFFER["distance_km"].append(distance_km)
    _REFERENCE_BUFFER["weight_kg"].append(weight_kg)
    _REFERENCE_BUFFER["predicted_minutes"].append(predicted_minutes)

    logger.debug("Logged prediction id=%d minutes=%.1f", pred.id, predicted_minutes)
    return pred


def log_anomaly(
    db: Session,
    building_id: str,
    timestamp: datetime,
    consumption_kwh: float,
    anomaly_score: float,
    is_anomaly: int,
    severity: str,
) -> AnomalyLog:
    """Persist an anomaly record for a building."""
    entry = AnomalyLog(
        building_id=building_id,
        timestamp=timestamp,
        consumption_kwh=consumption_kwh,
        anomaly_score=anomaly_score,
        is_anomaly=is_anomaly,
        severity=severity,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    _anomaly_flags_buffer.append(bool(is_anomaly))
    return entry


def get_anomaly_stats(db: Session) -> dict[str, Any]:
    """Return aggregate anomaly statistics from the database."""
    total = db.query(AnomalyLog).count()
    if total == 0:
        return {"total_anomalies": 0, "anomaly_rate": 0.0, "critical_count": 0}
    critical = db.query(AnomalyLog).filter(AnomalyLog.severity == "critical").count()
    anomalous = db.query(AnomalyLog).filter(AnomalyLog.is_anomaly == 1).count()
    return {
        "total_anomalies": total,
        "anomaly_rate": anomalous / total if total > 0 else 0.0,
        "critical_count": critical,
    }


# ---------------------------------------------------------------------------
# Legacy helpers (logistics domain)
# ---------------------------------------------------------------------------

def run_drift_check(db: Session, current_window: int = 100) -> dict[str, Any]:
    """Compare latest predictions against reference buffer; log results."""
    recent = (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .limit(current_window)
        .all()
    )
    if not recent:
        return {"status": "no_predictions", "features": {}}

    results: dict[str, Any] = {}
    for feature in ("distance_km", "weight_kg", "predicted_minutes"):
        ref = list(_REFERENCE_BUFFER[feature])
        cur = [getattr(r, feature) for r in recent]
        drift = compute_drift(ref, cur)
        results[feature] = drift

        if drift.get("drift_detected"):
            log = DriftLog(
                feature=feature,
                ks_statistic=drift["ks_statistic"],
                p_value=drift["p_value"],
                drift_detected=1,
            )
            db.add(log)
            logger.warning(
                "Drift detected in '%s': KS=%.4f p=%.4f",
                feature,
                drift["ks_statistic"],
                drift["p_value"],
            )

    db.commit()
    return {"status": "ok", "features": results}


def seed_reference_buffer(samples: list[dict]) -> None:
    """Pre-load reference buffer from training data summary."""
    for s in samples:
        for k in _REFERENCE_BUFFER:
            if k in s:
                _REFERENCE_BUFFER[k].append(s[k])
    logger.info("Reference buffer seeded with %d samples", len(samples))
