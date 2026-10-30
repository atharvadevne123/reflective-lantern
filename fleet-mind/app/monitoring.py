"""KS-test drift detection and prediction logging."""

from __future__ import annotations

import json
import logging
from collections import deque
from typing import Any

import numpy as np
from scipy import stats
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import DriftReport, VehiclePrediction
from app.features import RAW_FEATURES

logger = logging.getLogger(__name__)

# In-memory circular buffer for recent feature observations
_feature_buffer: dict[str, deque[float]] = {f: deque(maxlen=1000) for f in RAW_FEATURES}
# Baseline distributions loaded from training data
_baseline_stats: dict[str, np.ndarray] = {}


def _load_baseline() -> None:
    """Populate baseline statistics from a simple synthetic distribution."""
    rng = np.random.default_rng(0)
    _baseline_stats["speed_kmh"] = rng.uniform(0, 130, 2000)
    _baseline_stats["engine_temp_c"] = rng.uniform(60, 100, 2000)
    _baseline_stats["oil_pressure_bar"] = rng.uniform(1.5, 5.5, 2000)
    _baseline_stats["fuel_consumption_lph"] = rng.uniform(4, 12, 2000)
    _baseline_stats["battery_voltage_v"] = rng.uniform(11.5, 13.5, 2000)
    _baseline_stats["brake_pad_mm"] = rng.uniform(2, 12, 2000)
    _baseline_stats["tire_pressure_bar"] = rng.uniform(1.8, 2.8, 2000)
    _baseline_stats["rpm"] = rng.uniform(700, 4500, 2000)
    _baseline_stats["coolant_temp_c"] = rng.uniform(70, 95, 2000)
    _baseline_stats["transmission_temp_c"] = rng.uniform(60, 100, 2000)


_load_baseline()


def log_prediction(
    db: Session,
    vehicle_id: str,
    reading: dict[str, float],
    result: dict[str, Any],
) -> None:
    """Persist prediction to database and update in-memory feature buffer."""
    record = VehiclePrediction(
        vehicle_id=vehicle_id,
        speed_kmh=reading["speed_kmh"],
        engine_temp_c=reading["engine_temp_c"],
        oil_pressure_bar=reading["oil_pressure_bar"],
        fuel_consumption_lph=reading["fuel_consumption_lph"],
        battery_voltage_v=reading["battery_voltage_v"],
        brake_pad_mm=reading["brake_pad_mm"],
        tire_pressure_bar=reading["tire_pressure_bar"],
        rpm=reading["rpm"],
        coolant_temp_c=reading["coolant_temp_c"],
        transmission_temp_c=reading["transmission_temp_c"],
        maintenance_needed=result["maintenance_needed"],
        anomaly_score=result["anomaly_score"],
        anomaly_type=result.get("anomaly_type"),
        remaining_useful_life_km=result.get("remaining_useful_life_km"),
        confidence=result["confidence"],
        model_version=result["model_version"],
    )
    try:
        db.add(record)
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("Failed to persist prediction: %s", exc)

    for feature in RAW_FEATURES:
        if feature in reading:
            _feature_buffer[feature].append(reading[feature])


def run_drift_check(db: Session) -> list[dict[str, Any]]:
    """Run KS-test on buffered observations vs baseline. Persists and returns results."""
    settings = get_settings()
    reports: list[dict[str, Any]] = []

    for feature in RAW_FEATURES:
        observed = list(_feature_buffer[feature])
        if len(observed) < 50:
            continue

        baseline = _baseline_stats.get(feature)
        if baseline is None or len(baseline) == 0:
            continue

        ks_stat, p_value = stats.ks_2samp(baseline, observed)
        drift_detected = p_value < settings.drift_threshold

        report_obj = DriftReport(
            feature_name=feature,
            ks_statistic=float(ks_stat),
            p_value=float(p_value),
            drift_detected=drift_detected,
            sample_size=len(observed),
            details=json.dumps({"mean": float(np.mean(observed)), "std": float(np.std(observed))}),
        )
        try:
            db.add(report_obj)
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.error("Failed to save drift report: %s", exc)

        reports.append({
            "feature": feature,
            "ks_statistic": round(ks_stat, 6),
            "p_value": round(float(p_value), 6),
            "drift_detected": drift_detected,
            "sample_size": len(observed),
        })
        if drift_detected:
            logger.warning("Drift detected in %s: KS=%.4f p=%.6f", feature, ks_stat, p_value)

    return reports


def maintenance_rate(db: Session, window: int = 500) -> float:
    """Return fraction of recent predictions flagging maintenance."""
    try:
        recent = (
            db.query(VehiclePrediction.maintenance_needed)
            .order_by(VehiclePrediction.id.desc())
            .limit(window)
            .all()
        )
        if not recent:
            return 0.0
        return sum(1 for (r,) in recent if r) / len(recent)
    except Exception as exc:
        logger.error("Failed to compute maintenance rate: %s", exc)
        return 0.0
