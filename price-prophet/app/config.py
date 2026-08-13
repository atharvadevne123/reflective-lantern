"""
Application configuration for Price-Prophet.

Uses Pydantic BaseSettings so every field can be overridden via
environment variables (upper-cased field name, e.g. ``DEBUG=true``).
"""

try:
    from pydantic_settings import BaseSettings
except ImportError:  # pragma: no cover - pydantic v1 fallback
    from pydantic import BaseSettings  # type: ignore[no-redef]


class Settings(BaseSettings):
    """Central settings object.  All values are readable from the
    environment, which makes twelve-factor configuration trivial."""

    app_name: str = "Price-Prophet"
    debug: bool = False

    # Model persistence
    model_dir: str = "models"

    # Cache behaviour
    cache_ttl_seconds: int = 300

    # Pricing guardrails
    max_price_multiplier: float = 5.0
    min_price_multiplier: float = 0.5

    # Demand-elasticity default when we cannot estimate from data
    default_elasticity: float = -1.5

    # Logging
    log_level: str = "INFO"

    class Config:
        env_prefix = ""
        case_sensitive = False


# Module-level singleton - import this everywhere instead of
# instantiating Settings directly.
settings = Settings()
