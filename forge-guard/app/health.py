"""Health check helpers for the Forge-Guard liveness and readiness probes."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def check_model_loaded() -> dict[str, Any]:
    """Return model health status without triggering a full load.

    Reads MODEL_PATH from the environment and checks whether the file exists
    and is non-empty.  Does NOT deserialise the model — keeping this fast
    enough for a 200ms liveness check.
    """
    model_path = Path(os.getenv("MODEL_PATH", "model.joblib"))
    exists = model_path.exists()
    size_bytes = model_path.stat().st_size if exists else 0
    return {
        "model_file_exists": exists,
        "model_file_size_bytes": size_bytes,
        "ok": exists and size_bytes > 0,
    }


def check_database_reachable(db_url: str | None = None) -> dict[str, Any]:
    """Probe database connectivity with a minimal SELECT 1 query.

    Args:
        db_url: SQLAlchemy connection string.  Falls back to DATABASE_URL
            environment variable and then the default SQLite path.

    Returns:
        Dict with ``reachable`` bool and optional ``error`` string.
    """
    url = db_url or os.getenv("DATABASE_URL", "sqlite:///./forge_guard.db")
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"reachable": True, "url_scheme": url.split("://")[0]}
    except Exception as exc:
        logger.warning("Database health check failed: %s", exc)
        return {"reachable": False, "error": str(exc)}


def composite_health() -> dict[str, Any]:
    """Aggregate model and database health into a single status dict.

    Returns:
        Dict with top-level ``status`` ("healthy" or "degraded"), plus
        ``model`` and ``database`` sub-dicts from each individual check.
    """
    model_status = check_model_loaded()
    db_status = check_database_reachable()
    healthy = model_status["ok"] and db_status["reachable"]
    return {
        "status": "healthy" if healthy else "degraded",
        "model": model_status,
        "database": db_status,
    }
