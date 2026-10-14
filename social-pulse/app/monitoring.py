"""KS-test drift detection and prediction logging."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from scipy import stats
from sqlalchemy.orm import Session

from app.database import DriftRecord, Prediction

logger = logging.getLogger(__name__)

DRIFT_THRESHOLD: float = 0.05  # p-value below which drift is flagged
MIN_SAMPLE_SIZE: int = 50  # minimum predictions before running KS test


def log_prediction(
    db: Session,
    request_id: str,
    sample: dict[str, Any],
    viral_probability: float,
    viral_label: bool,
    ensemble_confidence: float,
    model_version: str,
    latency_ms: float,
) -> Prediction:
    """Persist a prediction to the database.

    Args:
        db: Active SQLAlchemy session.
        request_id: Unique identifier for this inference request.
        sample: Raw feature dict from the request.
        viral_probability: Model output probability (0–1).
        viral_label: Thresholded binary prediction.
        ensemble_confidence: Distance from decision boundary (0–1).
        model_version: Version string of the serving model.
        latency_ms: Wall-clock inference time in milliseconds.

    Returns:
        The persisted Prediction ORM record.
    """
    record = Prediction(
        request_id=request_id,
        text=sample.get("text", ""),
        author_followers=int(sample.get("author_followers", 0)),
        has_media=bool(sample.get("has_media", False)),
        has_link=bool(sample.get("has_link", False)),
        hashtag_count=int(sample.get("hashtag_count", 0)),
        mention_count=int(sample.get("mention_count", 0)),
        hour_of_day=int(sample.get("hour_of_day", 12)),
        day_of_week=int(sample.get("day_of_week", 0)),
        viral_probability=viral_probability,
        viral_label=viral_label,
        ensemble_confidence=ensemble_confidence,
        model_version=model_version,
        latency_ms=latency_ms,
        feature_vector={k: v for k, v in sample.items() if k != "text"},
    )
    try:
        db.add(record)
        db.commit()
        db.refresh(record)
        logger.debug("Logged prediction %s (p=%.3f)", request_id, viral_probability)
    except Exception as exc:
        db.rollback()
        logger.error("Failed to log prediction %s: %s", request_id, exc)
        raise
    return record


def run_drift_check(
    db: Session,
    reference_window_days: int = 30,
    current_window_days: int = 7,
) -> list[DriftRecord]:
    """Run KS-test drift detection across numeric features.

    Compares the distribution of recent predictions against an older
    reference window.  A p-value below DRIFT_THRESHOLD flags the feature
    as drifted and writes a DriftRecord to the database.

    Args:
        db: Active SQLAlchemy session.
        reference_window_days: How many days back to use as reference.
        current_window_days: How many days of recent data to test.

    Returns:
        List of DriftRecord instances written to the database.
    """
    now = datetime.utcnow()
    ref_start = now - timedelta(days=reference_window_days)
    cur_start = now - timedelta(days=current_window_days)

    ref_rows = (
        db.query(Prediction)
        .filter(Prediction.created_at >= ref_start, Prediction.created_at < cur_start)
        .all()
    )
    cur_rows = (
        db.query(Prediction)
        .filter(Prediction.created_at >= cur_start)
        .all()
    )

    records: list[DriftRecord] = []

    if len(ref_rows) < MIN_SAMPLE_SIZE or len(cur_rows) < MIN_SAMPLE_SIZE:
        logger.info(
            "Skipping drift check: ref=%d cur=%d (need %d each)",
            len(ref_rows),
            len(cur_rows),
            MIN_SAMPLE_SIZE,
        )
        return records

    feature_extractors: dict[str, Any] = {
        "viral_probability": lambda r: r.viral_probability,
        "ensemble_confidence": lambda r: r.ensemble_confidence,
        "author_followers_log": lambda r: np.log1p(r.author_followers),
        "hashtag_count": lambda r: float(r.hashtag_count),
        "mention_count": lambda r: float(r.mention_count),
        "hour_of_day": lambda r: float(r.hour_of_day),
    }

    for feature_name, extractor in feature_extractors.items():
        try:
            ref_vals = np.array([extractor(r) for r in ref_rows], dtype=float)
            cur_vals = np.array([extractor(r) for r in cur_rows], dtype=float)
            ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
            drift_detected = bool(p_value < DRIFT_THRESHOLD)

            drift_record = DriftRecord(
                feature_name=feature_name,
                ks_statistic=float(ks_stat),
                p_value=float(p_value),
                drift_detected=drift_detected,
                sample_size=len(cur_vals),
                checked_at=now,
            )
            db.add(drift_record)
            records.append(drift_record)

            if drift_detected:
                logger.warning(
                    "DRIFT detected on '%s': KS=%.4f, p=%.4f",
                    feature_name,
                    ks_stat,
                    p_value,
                )
            else:
                logger.info(
                    "No drift on '%s': KS=%.4f, p=%.4f",
                    feature_name,
                    ks_stat,
                    p_value,
                )
        except Exception as exc:
            logger.error("Drift check failed for '%s': %s", feature_name, exc)

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("Failed to persist drift records: %s", exc)
        raise

    n_drifted = sum(1 for r in records if r.drift_detected)
    logger.info("Drift check complete: %d/%d features drifted", n_drifted, len(records))
    return records


def get_drift_summary(db: Session, limit: int = 20) -> list[dict[str, Any]]:
    """Return recent drift check results as a list of dicts.

    Args:
        db: Active SQLAlchemy session.
        limit: Maximum number of records to return.

    Returns:
        List of dicts suitable for JSON serialisation.
    """
    rows = (
        db.query(DriftRecord)
        .order_by(DriftRecord.checked_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "feature_name": r.feature_name,
            "ks_statistic": r.ks_statistic,
            "p_value": r.p_value,
            "drift_detected": r.drift_detected,
            "sample_size": r.sample_size,
            "checked_at": r.checked_at.isoformat() if r.checked_at else None,
        }
        for r in rows
    ]


def should_retrain(db: Session, min_new_samples: int = 1000) -> bool:
    """Return True if enough new predictions have accumulated for retraining.

    Args:
        db: Active SQLAlchemy session.
        min_new_samples: Minimum number of unlabelled predictions needed.

    Returns:
        True when retraining is recommended.
    """
    count = db.query(Prediction).count()
    return count >= min_new_samples
