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
    """

    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./watt_guard.db")
    )
    model_path: str = field(default_factory=lambda: os.getenv("MODEL_PATH", "model.joblib"))
    metrics_path: str = field(default_factory=lambda: os.getenv("METRICS_PATH", "metrics.json"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    rate_limit_per_minute: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
    )


settings = Settings()
