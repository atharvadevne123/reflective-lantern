"""Model monitoring: drift detection and prediction logging."""
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import TicketPrediction

logger = logging.getLogger(__name__)

MIN_DRIFT_SAMPLES = 50


def log_prediction(
    db: Session,
    ticket_id: str,
    subject: str,
    description: str,
    prediction: dict[str, Any],
) -> None:
    """Persist a prediction record for monitoring."""
    record = TicketPrediction(
        ticket_id=ticket_id,
        subject=subject,
        description=description,
        predicted_priority=prediction["priority"],
        predicted_category=prediction.get("category", "unknown"),
        predicted_resolution_hours=prediction["resolution_hours_estimate"],
        priority_confidence=prediction["confidence"],
        subject_len=len(subject),
        desc_len=len(description),
        word_count=len((subject + " " + description).split()),
        has_error_keyword=int(any(kw in (subject + description).lower() for kw in ["error", "exception", "500", "crash"])),
        has_urgent_keyword=int(any(kw in (subject + description).lower() for kw in ["urgent", "critical", "emergency", "down"])),
    )
    db.add(record)
    db.commit()
    logger.debug("Logged prediction for ticket %s", ticket_id)


def compute_drift(reference: list[float], current: list[float]) -> dict[str, Any]:
    """Kolmogorov-Smirnov drift test between reference and current distributions."""
    if len(reference) < MIN_DRIFT_SAMPLES or len(current) < MIN_DRIFT_SAMPLES:
        return {
            "ks_statistic": None,
            "p_value": None,
            "drift_detected": False,
            "reason": "insufficient_samples",
            "reference_n": len(reference),
            "current_n": len(current),
        }
    stat, p = ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < 0.05),
        "reference_n": len(reference),
        "current_n": len(current),
    }


def get_drift_report(db: Session, window_days: int = 7) -> dict[str, Any]:
    """Compute drift between historical baseline and recent predictions."""
    cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=window_days)

    all_records = db.query(TicketPrediction).all()
    recent_records = [r for r in all_records if r.created_at >= cutoff]
    old_records = [r for r in all_records if r.created_at < cutoff]

    if not all_records:
        return {"status": "no_data", "total_predictions": 0}

    def extract(records: list, attr: str) -> list[float]:
        return [getattr(r, attr) for r in records if getattr(r, attr) is not None]

    drift_confidence = compute_drift(
        extract(old_records, "priority_confidence"),
        extract(recent_records, "priority_confidence"),
    )
    drift_resolution = compute_drift(
        extract(old_records, "predicted_resolution_hours"),
        extract(recent_records, "predicted_resolution_hours"),
    )
    drift_word_count = compute_drift(
        extract(old_records, "word_count"),
        extract(recent_records, "word_count"),
    )

    priority_dist = {}
    for r in all_records:
        priority_dist[r.predicted_priority] = priority_dist.get(r.predicted_priority, 0) + 1

    return {
        "status": "ok",
        "total_predictions": len(all_records),
        "recent_predictions": len(recent_records),
        "window_days": window_days,
        "priority_distribution": priority_dist,
        "drift": {
            "confidence_score": drift_confidence,
            "resolution_hours": drift_resolution,
            "word_count": drift_word_count,
        },
        "any_drift_detected": any([
            drift_confidence.get("drift_detected"),
            drift_resolution.get("drift_detected"),
            drift_word_count.get("drift_detected"),
        ]),
    }
