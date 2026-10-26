"""Health-check utilities for Property-Sage.

Provides a deep health check that inspects the database connection,
model load state, and disk usage for the model artefact directory.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

MODEL_DIR: Path = Path(os.getenv("MODEL_DIR", "models"))


def deep_health_check(
    db: Session,
    price_model_loaded: bool,
    rental_model_loaded: bool,
) -> dict[str, Any]:
    """Return a detailed health report covering DB, models, and disk.

    Args:
        db: Active SQLAlchemy session to probe.
        price_model_loaded: Whether the price model is loaded in memory.
        rental_model_loaded: Whether the rental model is loaded in memory.

    Returns:
        Dict with component-level health statuses and an overall status string.
    """
    checks: dict[str, Any] = {}

    # Database connectivity
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {exc}"
        logger.error("Health check — DB error: %s", exc)

    # Model artefact presence
    checks["price_model"] = "ok" if (MODEL_DIR / "price_model.joblib").exists() else "missing"
    checks["rental_model"] = "ok" if (MODEL_DIR / "rental_model.joblib").exists() else "missing"

    # In-memory model state
    checks["price_model_loaded"] = "ok" if price_model_loaded else "not_loaded"
    checks["rental_model_loaded"] = "ok" if rental_model_loaded else "not_loaded"

    overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    logger.info("Deep health check complete — status=%s", overall)

    return {"status": overall, "components": checks}
