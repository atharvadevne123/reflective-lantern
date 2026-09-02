"""CRUD operations for Incidents, Predictions, and DriftAlerts."""

import logging
from datetime import datetime, timedelta

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


def get_incident(db: Session, incident_id: int) -> Incident | None:
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
    service_name: str | None = None,
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


def get_incidents_by_service(
    db: Session,
    service_name: str,
    limit: int = 50,
) -> list[Incident]:
    """Return incidents for a specific service, newest first.

    Args:
        db: Active SQLAlchemy session.
        service_name: Exact service name to filter by.
        limit: Maximum records to return.

    Returns:
        List of Incident objects.
    """
    try:
        return (
            db.query(Incident)
            .filter(Incident.service_name == service_name)
            .order_by(Incident.created_at.desc())
            .limit(limit)
            .all()
        )
    except Exception:
        logger.exception("Failed to get incidents for service=%s", service_name)
        raise


def delete_old_predictions(db: Session, older_than_days: int = 30) -> int:
    """Delete prediction records older than a given number of days.

    Args:
        db: Active SQLAlchemy session.
        older_than_days: Predictions older than this many days are removed.

    Returns:
        Number of rows deleted.
    """
    try:
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        deleted = (
            db.query(Prediction)
            .filter(Prediction.created_at < cutoff)
            .delete(synchronize_session=False)
        )
        db.commit()
        logger.info("Deleted %d old predictions (cutoff=%s)", deleted, cutoff.date())
        return deleted
    except Exception:
        db.rollback()
        logger.exception("Failed to delete old predictions")
        raise


def bulk_create_predictions(db: Session, items: list[dict]) -> int:
    """Insert multiple prediction records in a single transaction.

    Args:
        db: Active SQLAlchemy session.
        items: List of field dicts, one per Prediction row.

    Returns:
        Number of rows inserted.
    """
    try:
        db.bulk_insert_mappings(Prediction, items)
        db.commit()
        logger.info("Bulk inserted %d predictions", len(items))
        return len(items)
    except Exception:
        db.rollback()
        logger.exception("Failed to bulk insert predictions")
        raise


def get_prediction_by_id(db: Session, prediction_id: int) -> Prediction | None:
    """Fetch a single prediction by primary key.

    Args:
        db: Active SQLAlchemy session.
        prediction_id: Primary key to look up.

    Returns:
        Prediction instance or None if not found.
    """
    try:
        return db.query(Prediction).filter(Prediction.id == prediction_id).first()
    except Exception:
        logger.exception("Failed to fetch prediction id=%d", prediction_id)
        raise


def count_predictions_by_service(db: Session, service_name: str) -> int:
    """Count predictions for a specific service.

    Args:
        db: Active SQLAlchemy session.
        service_name: Service to filter by.

    Returns:
        Integer count of predictions for that service.
    """
    try:
        return (
            db.query(func.count(Prediction.id))
            .filter(Prediction.service_name == service_name)
            .scalar()
            or 0
        )
    except Exception:
        logger.exception("Failed to count predictions for service=%s", service_name)
        raise
