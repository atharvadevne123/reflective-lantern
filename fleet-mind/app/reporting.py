"""Report generation helpers for fleet maintenance summaries."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from app.database import DriftReport, VehiclePrediction

logger = logging.getLogger(__name__)


def fleet_summary(db: Session, limit: int = 1000) -> dict[str, Any]:
    """Return an aggregated fleet maintenance summary.

    Args:
        db: Active SQLAlchemy session.
        limit: Maximum number of recent predictions to include.

    Returns:
        Dict with total_predictions, maintenance_rate, top_anomaly_types,
        and avg_remaining_useful_life_km.
    """
    try:
        rows = (
            db.query(VehiclePrediction)
            .order_by(VehiclePrediction.id.desc())
            .limit(limit)
            .all()
        )
    except Exception as exc:
        logger.error("Failed to query predictions for summary: %s", exc)
        return {}

    if not rows:
        return {"total_predictions": 0, "maintenance_rate": 0.0}

    maintenance_count = sum(1 for r in rows if r.maintenance_needed)
    anomaly_counts: dict[str, int] = {}
    rul_values: list[float] = []

    for row in rows:
        if row.anomaly_type:
            anomaly_counts[row.anomaly_type] = anomaly_counts.get(row.anomaly_type, 0) + 1
        if row.remaining_useful_life_km is not None:
            rul_values.append(row.remaining_useful_life_km)

    top_anomalies = sorted(anomaly_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    avg_rul = sum(rul_values) / len(rul_values) if rul_values else None

    return {
        "total_predictions": len(rows),
        "maintenance_rate": round(maintenance_count / len(rows), 4),
        "top_anomaly_types": [{"type": t, "count": c} for t, c in top_anomalies],
        "avg_remaining_useful_life_km": round(avg_rul, 1) if avg_rul is not None else None,
    }


def recent_drift_reports(db: Session, limit: int = 20) -> list[dict[str, Any]]:
    """Return the most recent drift detection reports.

    Args:
        db: Active SQLAlchemy session.
        limit: Maximum rows to return.

    Returns:
        List of dicts describing each DriftReport record.
    """
    try:
        reports = (
            db.query(DriftReport)
            .order_by(DriftReport.id.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "feature": r.feature_name,
                "ks_statistic": r.ks_statistic,
                "p_value": r.p_value,
                "drift_detected": r.drift_detected,
                "sample_size": r.sample_size,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in reports
        ]
    except Exception as exc:
        logger.error("Failed to fetch drift reports: %s", exc)
        return []
