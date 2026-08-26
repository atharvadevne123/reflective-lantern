"""Centralised configuration loaded from the environment."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back on malformed values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a float; using default %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    """Read an int from the environment, falling back on malformed values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not an int; using default %s", name, raw, default)
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the Threat-Lens service."""

    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./threat_lens.db")
    )
    model_path: Path = field(default_factory=lambda: Path(os.getenv("MODEL_PATH", "model.joblib")))
    metrics_path: Path = field(
        default_factory=lambda: Path(os.getenv("METRICS_PATH", "metrics.json"))
    )
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: _env_int("API_PORT", 8000))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())
    drift_threshold: float = field(default_factory=lambda: _env_float("DRIFT_THRESHOLD", 0.05))
    reference_window: int = field(default_factory=lambda: _env_int("REFERENCE_WINDOW", 500))
    rate_limit_per_minute: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_PER_MINUTE", 120)
    )
    max_batch_size: int = field(default_factory=lambda: _env_int("MAX_BATCH_SIZE", 100))

    def is_postgres(self) -> bool:
        """Return True when configured against PostgreSQL rather than SQLite."""
        return self.database_url.startswith("postgresql")


def get_settings() -> Settings:
    """Build a Settings instance from the current environment."""
    return Settings()
