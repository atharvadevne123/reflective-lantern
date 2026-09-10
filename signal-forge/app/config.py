"""Application configuration loaded from environment variables."""

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    """Immutable application settings resolved from environment at startup."""

    database_url: str = field(
        default_factory=lambda: os.environ.get("DATABASE_URL", "sqlite:///./signal_forge.db")
    )
    model_path: str = field(
        default_factory=lambda: os.environ.get("MODEL_PATH", "/tmp/signal_forge_model.pkl")
    )
    faiss_index_path: str = field(
        default_factory=lambda: os.environ.get("FAISS_INDEX_PATH", "/tmp/signal_forge.faiss")
    )
    app_env: str = field(
        default_factory=lambda: os.environ.get("APP_ENV", "development")
    )
    log_level: str = field(
        default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO")
    )
    drift_ks_threshold: float = field(
        default_factory=lambda: float(os.environ.get("DRIFT_KS_THRESHOLD", "0.05"))
    )
    reference_window_days: int = field(
        default_factory=lambda: int(os.environ.get("REFERENCE_WINDOW_DAYS", "30"))
    )
    monitor_window_days: int = field(
        default_factory=lambda: int(os.environ.get("MONITOR_WINDOW_DAYS", "7"))
    )
    rate_limit_requests: int = field(
        default_factory=lambda: int(os.environ.get("RATE_LIMIT_REQUESTS", "60"))
    )
    rate_limit_window: float = field(
        default_factory=lambda: float(os.environ.get("RATE_LIMIT_WINDOW", "60.0"))
    )

    def __post_init__(self) -> None:
        logging.basicConfig(level=getattr(logging, self.log_level.upper(), logging.INFO))
        logger.info(
            "Settings loaded: env=%s db=%s",
            self.app_env,
            self.database_url.split("@")[-1] if "@" in self.database_url else self.database_url,
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the singleton Settings instance (lazy-initialised)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
