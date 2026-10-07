"""Drift detection and prediction logging for Sports-Oracle."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

import numpy as np
from scipy.stats import ks_2samp

from app.database import DriftLog, PredictionLog, get_session

logger = logging.getLogger(__name__)


def compute_drift(reference: list[float], current: list[float]) -> dict[str, Any]:
    """Run KS-test between reference and current distributions."""
    if len(reference) < 10 or len(current) < 10:
        return {
            "ks_statistic": 0.0,
            "p_value": 1.0,
            "drift_detected": False,
            "warning": "insufficient samples",
        }
    stat, p = ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < 0.05),
    }


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between reference and current."""
    ref_hist, edges = np.histogram(reference, bins=bins, range=(reference.min(), reference.max()))
    cur_hist, _ = np.histogram(current, bins=edges)
    ref_pct = (ref_hist + 1e-6) / (len(reference) + 1e-6 * bins)
    cur_pct = (cur_hist + 1e-6) / (len(current) + 1e-6 * bins)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def check_feature_drift(
    reference_features: list[dict],
    current_features: list[dict],
    feature_names: list[str] | None = None,
) -> dict[str, dict]:
    """Check drift for each feature column and persist to DB."""
    if not reference_features or not current_features:
        return {}

    all_keys = feature_names or list(reference_features[0].keys())
    results: dict[str, dict] = {}

    session = get_session()
    try:
        for key in all_keys:
            ref_vals = [float(r.get(key, 0.0)) for r in reference_features if key in r]
            cur_vals = [float(c.get(key, 0.0)) for c in current_features if key in c]
            if not ref_vals or not cur_vals:
                continue
            drift = compute_drift(ref_vals, cur_vals)
            results[key] = drift

            log = DriftLog(
                feature_name=key,
                ks_statistic=drift["ks_statistic"],
                p_value=drift["p_value"],
                drift_detected=drift["drift_detected"],
                window_size=len(cur_vals),
                checked_at=datetime.utcnow(),
            )
            session.add(log)

        session.commit()
        logger.info("drift_check_complete", extra={"n_features": len(results)})
    except Exception:
        session.rollback()
        logger.exception("drift_check_db_error")
    finally:
        session.close()

    return results


def log_prediction(
    home_team: str,
    away_team: str,
    competition: str,
    features: dict,
    prediction: dict,
    model_version: str = "1.0.0",
    latency_ms: float = 0.0,
    request_id: str | None = None,
) -> None:
    """Persist a prediction to the DB for monitoring."""
    session = get_session()
    try:
        rec = PredictionLog(
            request_id=request_id or str(uuid.uuid4()),
            home_team=home_team,
            away_team=away_team,
            competition=competition,
            features_json=features,
            predicted_outcome=prediction.get("predicted_outcome", ""),
            prob_home=prediction.get("prob_home", 0.0),
            prob_draw=prediction.get("prob_draw", 0.0),
            prob_away=prediction.get("prob_away", 0.0),
            confidence=prediction.get("confidence", 0.0),
            model_version=model_version,
            latency_ms=latency_ms,
            created_at=datetime.utcnow(),
        )
        session.add(rec)
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("log_prediction_error")
    finally:
        session.close()


def get_recent_predictions(n: int = 100) -> list[dict]:
    """Retrieve the most recent n predictions."""
    session = get_session()
    try:
        rows = (
            session.query(PredictionLog)
            .order_by(PredictionLog.created_at.desc())
            .limit(n)
            .all()
        )
        return [
            {
                "request_id": r.request_id,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "predicted_outcome": r.predicted_outcome,
                "prob_home": r.prob_home,
                "prob_draw": r.prob_draw,
                "prob_away": r.prob_away,
                "confidence": r.confidence,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
    except Exception:
        logger.exception("get_recent_predictions_error")
        return []
    finally:
        session.close()
