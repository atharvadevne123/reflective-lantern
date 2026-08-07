"""Centralised application settings read from environment variables.

All runtime configuration flows through :class:`Settings` so that the API,
the retraining pipeline, and the tests share a single source of truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    """Immutable application settings resolved at import time.

    Attributes:
        database_url: SQLAlchemy connection string (SQLite dev, PostgreSQL prod).
        model_path: Filesystem path of the persisted ensemble.
        metrics_path: Filesystem path of the training metrics JSON.
        log_level: Root logging level name.
        rate_limit_per_minute: Maximum requests per client per minute.
        db_pool_size: Number of persistent database connections in the pool.
        db_max_overflow: Extra connections allowed beyond pool_size.
        db_pool_recycle: Seconds before a connection is recycled to avoid timeouts.
        cors_origins: Comma-separated list of allowed CORS origins.
    """

    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./watt_guard.db"))
    model_path: str = field(default_factory=lambda: os.getenv("MODEL_PATH", "model.joblib"))
    metrics_path: str = field(default_factory=lambda: os.getenv("METRICS_PATH", "metrics.json"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    rate_limit_per_minute: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MINUTE", "120")))
    db_pool_size: int = field(default_factory=lambda: int(os.getenv("DB_POOL_SIZE", "5")))
    db_max_overflow: int = field(default_factory=lambda: int(os.getenv("DB_MAX_OVERFLOW", "10")))
    db_pool_recycle: int = field(default_factory=lambda: int(os.getenv("DB_POOL_RECYCLE", "1800")))
    cors_origins: str = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*"))


settings = Settings()

__all__ = [
    "Settings",
    "effective_log_level",
    "get_settings",
    "is_production",
    "settings",
]


def get_settings() -> Settings:
    """Return the module-level :class:`Settings` singleton.

    Returns:
        The application-wide Settings instance.
    """
    return settings


def is_production() -> bool:
    """Return True when DATABASE_URL points to a non-SQLite database.

    Useful for guarding prod-only code paths such as Prometheus export
    or strict rate-limiting.

    Returns:
        True when the database URL does not start with 'sqlite'.
    """
    return not settings.database_url.startswith("sqlite")


def effective_log_level() -> str:
    """Return the configured log level in upper-case form.

    Returns:
        Log level string such as 'INFO', 'DEBUG', 'WARNING'.
    """
    return settings.log_level.upper()
