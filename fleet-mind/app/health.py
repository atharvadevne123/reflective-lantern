"""Health and readiness check utilities."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def check_db(db: Session) -> bool:
    """Return True if the database is reachable, False otherwise."""
    try:
        import sqlalchemy

        db.execute(sqlalchemy.text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("DB health check failed: %s", exc)
        return False


def build_health_payload(model_loaded: bool, db: Session) -> dict[str, Any]:
    """Return a dict suitable for the /health response.

    Args:
        model_loaded: Whether the inference model is resident in memory.
        db: Active SQLAlchemy session used to verify DB connectivity.

    Returns:
        Dict with 'status', 'model_version', and 'db_ok' keys.
    """
    from app.model import MODEL_VERSION

    db_ok = check_db(db)
    status = "ok" if (model_loaded and db_ok) else "degraded"
    return {
        "status": status,
        "model_version": MODEL_VERSION,
        "db_ok": db_ok,
    }


def check_model(model: Any) -> bool:
    """Return True if model exposes predict_proba and appears healthy.

    Args:
        model: Inference model object to verify.

    Returns:
        True when the model has a callable predict_proba attribute.
    """
    return model is not None and callable(getattr(model, "predict_proba", None))


def readiness_summary(model_loaded: bool, db: Session, model: Any = None) -> dict[str, Any]:
    """Return a detailed readiness dict for the /ready endpoint.

    Args:
        model_loaded: Whether the model flag is set.
        db: Active SQLAlchemy session.
        model: Optional model object for deeper health check.

    Returns:
        Dict with 'ready', 'db_ok', 'model_ok', and 'issues' list.
    """
    db_ok = check_db(db)
    model_ok = model_loaded and check_model(model)
    issues = []
    if not db_ok:
        issues.append("database unreachable")
    if not model_ok:
        issues.append("model not loaded or invalid")
    return {
        "ready": db_ok and model_ok,
        "db_ok": db_ok,
        "model_ok": model_ok,
        "issues": issues,
    }
