"""Structured health-check helpers for Ticket-Oracle.

Provides a detailed liveness/readiness probe so orchestration layers
(Kubernetes, docker-compose healthcheck) can distinguish a cold-start
from a broken model or an unreachable database.
"""

from __future__ import annotations

import time
from typing import Any


def check_model_ready() -> dict[str, Any]:
    """Return model readiness status without side effects."""
    from app.model import MODEL_DIR

    priority_path = MODEL_DIR / "priority_model.joblib"
    sla_path = MODEL_DIR / "sla_model.joblib"
    ready = priority_path.exists() and sla_path.exists()
    return {
        "ready": ready,
        "priority_model_exists": priority_path.exists(),
        "sla_model_exists": sla_path.exists(),
    }


def check_db_reachable() -> dict[str, Any]:
    """Attempt a lightweight DB ping and report latency."""
    from app.database import _get_engine

    t0 = time.perf_counter()
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {"reachable": True, "latency_ms": latency_ms}
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}


def full_health() -> dict[str, Any]:
    """Aggregate health status for the /health endpoint."""
    model_status = check_model_ready()
    db_status = check_db_reachable()

    healthy = model_status["ready"] and db_status["reachable"]
    return {
        "status": "ok" if healthy else "degraded",
        "model": model_status,
        "database": db_status,
    }
