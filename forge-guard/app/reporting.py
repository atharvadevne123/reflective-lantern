"""Reporting and data export utilities for Forge-Guard.

Provides functions to export prediction logs and drift reports to CSV/JSON
for downstream analysis, dashboards, or compliance auditing.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from app.database import DriftReport, PredictionLog

logger = logging.getLogger(__name__)


def export_predictions_csv(
    db: Session,
    hours: int = 24,
    limit: int = 10000,
) -> str:
    """Export recent prediction logs to a CSV string.

    Args:
        db: Active SQLAlchemy session.
        hours: Look-back window in hours.
        limit: Maximum number of rows to export.

    Returns:
        A UTF-8 CSV string with header row included.
    """
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    logs = (
        db.query(PredictionLog)
        .filter(PredictionLog.timestamp >= cutoff)
        .order_by(PredictionLog.timestamp.desc())
        .limit(limit)
        .all()
    )
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "id",
            "timestamp",
            "temperature",
            "pressure",
            "vibration",
            "cycle_time",
            "tool_wear",
            "power_consumption",
            "humidity",
            "prediction",
            "defect_probability",
            "model_version",
            "correlation_id",
        ]
    )
    for row in logs:
        writer.writerow(
            [
                row.id,
                row.timestamp.isoformat(),
                row.temperature,
                row.pressure,
                row.vibration,
                row.cycle_time,
                row.tool_wear,
                row.power_consumption,
                row.humidity,
                row.prediction,
                row.defect_probability,
                row.model_version,
                row.correlation_id,
            ]
        )
    return buf.getvalue()


def export_drift_reports_json(
    db: Session,
    hours: int = 24,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """Export recent drift reports as a list of dicts.

    Args:
        db: Active SQLAlchemy session.
        hours: Look-back window in hours.
        limit: Maximum number of rows to export.

    Returns:
        List of drift report dicts sorted newest-first.
    """
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    reports = (
        db.query(DriftReport)
        .filter(DriftReport.timestamp >= cutoff)
        .order_by(DriftReport.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "feature": r.feature,
            "ks_statistic": r.ks_statistic,
            "p_value": r.p_value,
            "drift_detected": r.drift_detected,
            "window_size": r.window_size,
            "model_version": r.model_version,
        }
        for r in reports
    ]


def prediction_summary_json(
    db: Session,
    hours: int = 24,
) -> dict[str, Any]:
    """Return a summary dict suitable for JSON serialisation.

    Args:
        db: Active SQLAlchemy session.
        hours: Look-back window in hours.

    Returns:
        Dict with total, defect_count, defect_rate, and window_hours.
    """
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    logs = db.query(PredictionLog).filter(PredictionLog.timestamp >= cutoff).all()
    total = len(logs)
    defects = sum(1 for r in logs if r.prediction == 1)
    return {
        "window_hours": hours,
        "total": total,
        "defect_count": defects,
        "defect_rate": round(defects / total, 4) if total else 0.0,
        "exported_at": datetime.utcnow().isoformat(),
    }
