"""Model monitoring: drift detection and prediction logging."""
from __future__ import annotations

import logging

import numpy as np
from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import DriftLog, PredictionLog

logger = logging.getLogger(__name__)


def compute_drift(reference: list[float], current: list[float]) -> dict:
    """KS-test between reference and current feature distribution."""
    if len(reference) < 2 or len(current) < 2:
        return {"ks_statistic": 0.0, "p_value": 1.0, "drift_detected": False, "error": "insufficient_data"}
    stat, p = ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < 0.05),
    }


def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two distributions."""
    bins = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()) + 1e-9,
        n_bins + 1,
    )
    ref_counts = np.histogram(reference, bins=bins)[0] + 1e-6
    cur_counts = np.histogram(current, bins=bins)[0] + 1e-6
    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 4)


def log_drift(
    db: Session,
    feature_name: str,
    ks_statistic: float,
    p_value: float,
    drift_detected: bool,
) -> None:
    entry = DriftLog(
        feature_name=feature_name,
        ks_statistic=ks_statistic,
        p_value=p_value,
        drift_detected=drift_detected,
    )
    db.add(entry)
    db.commit()
    if drift_detected:
        logger.warning("Drift detected on '%s': KS=%.4f, p=%.4f", feature_name, ks_statistic, p_value)


def log_prediction(
    db: Session,
    match_id: str,
    home_team: str,
    away_team: str,
    predicted_outcome: str,
    home_win_prob: float,
    draw_prob: float,
    away_win_prob: float,
    confidence: float,
    model_version: str = "1.0.0",
) -> None:
    entry = PredictionLog(
        match_id=match_id,
        home_team=home_team,
        away_team=away_team,
        predicted_outcome=predicted_outcome,
        home_win_prob=home_win_prob,
        draw_prob=draw_prob,
        away_win_prob=away_win_prob,
        confidence=confidence,
        model_version=model_version,
    )
    db.add(entry)
    db.commit()
    logger.info(
        "Prediction logged: %s vs %s -> %s (conf=%.3f)",
        home_team, away_team, predicted_outcome, confidence,
    )


def get_drift_summary(db: Session, limit: int = 50) -> list[dict]:
    rows = db.query(DriftLog).order_by(DriftLog.checked_at.desc()).limit(limit).all()
    return [
        {
            "feature": r.feature_name,
            "ks_statistic": r.ks_statistic,
            "p_value": r.p_value,
            "drift_detected": r.drift_detected,
            "checked_at": r.checked_at.isoformat() if r.checked_at else None,
        }
        for r in rows
    ]


def get_prediction_metrics(db: Session) -> dict:
    total = db.query(PredictionLog).count()
    outcomes: dict[str, int] = {}
    for row in db.query(PredictionLog).all():
        outcomes[row.predicted_outcome] = outcomes.get(row.predicted_outcome, 0) + 1
    return {
        "total_predictions": total,
        "outcome_distribution": outcomes,
    }
