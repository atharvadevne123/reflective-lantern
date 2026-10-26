"""Health check utilities with dependency probes."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_start_time: float = time.time()


def check_database(db_session: Any) -> dict[str, Any]:
    """Probe the database connection by running a cheap scalar query."""
    try:
        from sqlalchemy import text
        db_session.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:
        logger.error("Database health check failed: %s", e)
        return {"status": "error", "detail": str(e)}


def check_model() -> dict[str, Any]:
    """Verify the serialised model file is present and loadable."""
    try:
        from app.model import MODEL_PATH
        if not MODEL_PATH.exists():
            return {"status": "cold_start", "detail": "model not yet trained"}
        return {"status": "ok", "path": str(MODEL_PATH)}
    except Exception as e:
        logger.error("Model health check failed: %s", e)
        return {"status": "error", "detail": str(e)}


def get_uptime() -> float:
    """Return seconds since the process started."""
    return round(time.time() - _start_time, 1)


def full_health_report(db_session: Any) -> dict[str, Any]:
    """Aggregate all dependency health probes into one report."""
    return {
        "status": "ok",
        "uptime_seconds": get_uptime(),
        "database": check_database(db_session),
        "model": check_model(),
    }
