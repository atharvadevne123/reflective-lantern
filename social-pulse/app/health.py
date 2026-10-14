"""Health and readiness check helpers for Social-Pulse."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_start_time: float = time.time()


@dataclass
class HealthStatus:
    """Aggregated health check result for the service.

    Attributes:
        status: Overall status string — "ok" or "degraded".
        model_loaded: Whether the ML model is ready for inference.
        db_reachable: Whether the database is reachable.
        model_version: Current model version string.
        uptime_seconds: Seconds since the process started.
        checks: Dict of individual check names to bool outcomes.
    """

    status: str
    model_loaded: bool
    db_reachable: bool
    model_version: str
    uptime_seconds: float
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "status": self.status,
            "model_loaded": self.model_loaded,
            "db_reachable": self.db_reachable,
            "model_version": self.model_version,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "checks": self.checks,
        }


def check_db(db: Session) -> bool:
    """Execute a trivial query to verify the database is reachable.

    Args:
        db: Active SQLAlchemy session.

    Returns:
        True if the query succeeded, False otherwise.
    """
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("DB health check failed: %s", exc)
        return False


def build_health_status(
    db: Session,
    model_loaded: bool,
    model_version: str,
) -> HealthStatus:
    """Build a HealthStatus by running all individual checks.

    Args:
        db: Active SQLAlchemy session.
        model_loaded: Whether the ML model is available.
        model_version: Current model version string.

    Returns:
        A populated HealthStatus instance.
    """
    db_ok = check_db(db)
    checks = {
        "model": model_loaded,
        "database": db_ok,
    }
    overall = "ok" if all(checks.values()) else "degraded"
    return HealthStatus(
        status=overall,
        model_loaded=model_loaded,
        db_reachable=db_ok,
        model_version=model_version,
        uptime_seconds=time.time() - _start_time,
        checks=checks,
    )
