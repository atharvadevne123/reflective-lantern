"""Centralised configuration for Cyber-Guard, loaded from the environment."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back on a malformed value.

    Args:
        name: Environment variable name.
        default: Value used when unset or unparseable.

    Returns:
        The parsed float, or ``default``.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("invalid float for %s=%r, using default %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    """Read an int from the environment, falling back on a malformed value.

    Args:
        name: Environment variable name.
        default: Value used when unset or unparseable.

    Returns:
        The parsed int, or ``default``.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("invalid int for %s=%r, using default %s", name, raw, default)
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings resolved from environment variables."""

    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./cyber_guard.db")
    )
    model_path: str = field(default_factory=lambda: os.getenv("MODEL_PATH", "model.joblib"))
    metrics_path: str = field(default_factory=lambda: os.getenv("METRICS_PATH", "metrics.json"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    api_version: str = field(default_factory=lambda: os.getenv("API_VERSION", "1.0.0"))
    drift_p_threshold: float = field(default_factory=lambda: _env_float("DRIFT_P_THRESHOLD", 0.05))
    reference_window_days: int = field(default_factory=lambda: _env_int("REFERENCE_WINDOW_DAYS", 7))
    rate_limit_per_minute: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_PER_MINUTE", 120)
    )
    retrain_accuracy_floor: float = field(
        default_factory=lambda: _env_float("RETRAIN_ACCURACY_FLOOR", 0.70)
    )

    def is_sqlite(self) -> bool:
        """Return True when the configured database is SQLite."""
        return self.database_url.startswith("sqlite")


def get_settings() -> Settings:
    """Build a Settings instance from the current environment.

    Returns:
        A frozen Settings snapshot. Not cached, so tests can change the
        environment and pick up the new values.
    """
    return Settings()
