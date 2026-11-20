"""CRUD operations for Incidents, Predictions, and DriftAlerts."""

import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import DriftAlert, Incident, Prediction

logger = logging.getLogger(__name__)


def create_incident(db: Session, data: dict) -> Incident:
    """Insert a new incident record and return it.

    Args:
        db: Active SQLAlchemy session.
        data: Dict of field values matching the Incident model.

    Returns:
        Newly created Incident with populated id.
    """
    try:
        incident = Incident(**data)
        db.add(incident)
        db.commit()
        db.refresh(incident)
        logger.info("Created incident id=%d service=%s", incident.id, incident.service_name)
        return incident
    except Exception:
        db.rollback()
        logger.exception("Failed to create incident")
        raise


def get_incident(db: Session, incident_id: int) -> Optional[Incident]:
    """Fetch a single incident by primary key.

    Args:
        db: Active SQLAlchemy session.
        incident_id: Primary key to look up.

    Returns:
        Incident instance or None if not found.
    """
    try:
        return db.query(Incident).filter(Incident.id == incident_id).first()
    except Exception:
        logger.exception("Failed to fetch incident id=%d", incident_id)
        raise


def list_incidents(
    db: Session,
    limit: int = 100,
    offset: int = 0,
    service_name: Optional[str] = None,
) -> list[Incident]:
    """List incidents with optional service filter and pagination.

    Args:
        db: Active SQLAlchemy session.
        limit: Maximum number of rows to return.
        offset: Number of rows to skip.
        service_name: If provided, filter by this service.

    Returns:
        List of Incident objects.
    """
    try:
        q = db.query(Incident)
        if service_name:
            q = q.filter(Incident.service_name == service_name)
        return q.order_by(Incident.created_at.desc()).limit(limit).offset(offset).all()
    except Exception:
        logger.exception("Failed to list incidents")
        raise


def create_prediction(db: Session, data: dict) -> Prediction:
    """Insert a new prediction record.

    Args:
        db: Active SQLAlchemy session.
        data: Dict of field values matching the Prediction model.

    Returns:
        Newly created Prediction with populated id.
    """
    try:
        prediction = Prediction(**data)
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        logger.info("Created prediction id=%d", prediction.id)
        return prediction
    except Exception:
        db.rollback()
        logger.exception("Failed to create prediction")
        raise


def count_predictions(db: Session) -> int:
    """Return total number of prediction records.

    Args:
        db: Active SQLAlchemy session.

    Returns:
        Integer count.
    """
    try:
        return db.query(func.count(Prediction.id)).scalar() or 0
    except Exception:
        logger.exception("Failed to count predictions")
        raise


def count_incidents_predicted(db: Session) -> int:
    """Count predictions flagged as incidents.

    Args:
        db: Active SQLAlchemy session.

    Returns:
        Integer count of predicted incidents.
    """
    try:
        return (
            db.query(func.count(Prediction.id))
            .filter(Prediction.predicted_incident.is_(True))
            .scalar()
            or 0
        )
    except Exception:
        logger.exception("Failed to count predicted incidents")
        raise


def avg_confidence(db: Session) -> float:
    """Return average prediction confidence score.

    Args:
        db: Active SQLAlchemy session.

    Returns:
        Mean confidence as float, or 0.0 if no records.
    """
    try:
        result = db.query(func.avg(Prediction.confidence)).scalar()
        return float(result) if result is not None else 0.0
    except Exception:
        logger.exception("Failed to compute avg confidence")
        raise


def create_drift_alert(db: Session, data: dict) -> DriftAlert:
    """Insert a drift alert record.

    Args:
        db: Active SQLAlchemy session.
        data: Dict of field values matching the DriftAlert model.

    Returns:
        Newly created DriftAlert.
    """
    try:
        alert = DriftAlert(**data)
        db.add(alert)
        db.commit()
        db.refresh(alert)
        logger.info(
            "Drift alert id=%d feature=%s drifted=%s",
            alert.id,
            alert.feature_name,
            alert.drifted,
        )
        return alert
    except Exception:
        db.rollback()
        logger.exception("Failed to create drift alert")
        raise


def count_drift_alerts_last_24h(db: Session) -> int:
    """Count drift alerts raised in the past 24 hours.

    Args:
        db: Active SQLAlchemy session.

    Returns:
        Integer count.
    """
    try:
        since = datetime.utcnow() - timedelta(hours=24)
        return (
            db.query(func.count(DriftAlert.id))
            .filter(DriftAlert.created_at >= since, DriftAlert.drifted.is_(True))
            .scalar()
            or 0
        )
    except Exception:
        logger.exception("Failed to count drift alerts")
        raise
