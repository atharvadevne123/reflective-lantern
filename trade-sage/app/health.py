"""Health check utilities for liveness and readiness probes."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field


@dataclass
class HealthStatus:
    """Container for service health information.

    Attributes:
        status: 'healthy', 'degraded', or 'unhealthy'.
        checks: Individual component check results.
        uptime_seconds: Seconds since process start.
    """

    status: str
    checks: dict[str, str] = field(default_factory=dict)
    uptime_seconds: float = 0.0

    def is_healthy(self) -> bool:
        return self.status == "healthy"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "checks": self.checks,
            "uptime_seconds": round(self.uptime_seconds, 1),
        }


_start_time = time.monotonic()


def check_database(db_url: str) -> str:
    """Return 'ok' if the DB URL is set, 'misconfigured' otherwise."""
    return "ok" if db_url else "misconfigured"


def check_model_file(model_path: str) -> str:
    """Return 'loaded' if the model file exists, 'missing' otherwise."""
    return "loaded" if os.path.exists(model_path) else "missing"


def build_health_status(db_url: str, model_path: str) -> HealthStatus:
    """Aggregate component checks into a HealthStatus.

    Args:
        db_url: Database connection string.
        model_path: Path to the persisted model artifact.

    Returns:
        HealthStatus with overall status and individual check results.
    """
    db_check = check_database(db_url)
    model_check = check_model_file(model_path)

    checks = {"database": db_check, "model": model_check}
    if all(v in ("ok", "loaded") for v in checks.values()):
        overall = "healthy"
    elif any(v in ("missing",) for v in checks.values()):
        overall = "degraded"
    else:
        overall = "unhealthy"

    return HealthStatus(
        status=overall,
        checks=checks,
        uptime_seconds=time.monotonic() - _start_time,
    )
