"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    """All configurable settings for Social-Pulse.

    Values are read from environment variables at import time, with
    sensible defaults for local development.
    """

    database_url: str = field(
        default_factory=lambda: os.environ.get("DATABASE_URL", "sqlite:///./social_pulse.db")
    )
    model_dir: str = field(
        default_factory=lambda: os.environ.get("MODEL_DIR", "/tmp/social_pulse_models")
    )
    model_version: str = field(default_factory=lambda: os.environ.get("MODEL_VERSION", "1.0.0"))
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO").upper())
    rate_limit: int = field(default_factory=lambda: int(os.environ.get("RATE_LIMIT", "60")))
    rate_window: int = field(default_factory=lambda: int(os.environ.get("RATE_WINDOW", "60")))
    drift_threshold: float = field(
        default_factory=lambda: float(os.environ.get("DRIFT_THRESHOLD", "0.05"))
    )
    min_retrain_samples: int = field(
        default_factory=lambda: int(os.environ.get("MIN_RETRAIN_SAMPLES", "1000"))
    )
    tfidf_max_features: int = field(
        default_factory=lambda: int(os.environ.get("TFIDF_MAX_FEATURES", "200"))
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the singleton Settings instance, creating it on first call.

    Returns:
        The global :class:`Settings` instance.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset the singleton (useful in tests to pick up env var changes)."""
    global _settings
    _settings = None
