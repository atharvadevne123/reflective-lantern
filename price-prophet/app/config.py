"""
Application configuration for Price-Prophet.

Uses Pydantic BaseSettings so every field can be overridden via
environment variables (upper-cased field name, e.g. ``DEBUG=true``).
"""

try:
    from pydantic_settings import BaseSettings
except ImportError:  # pragma: no cover - pydantic v1 fallback
    from pydantic import BaseSettings  # type: ignore[no-redef]

try:
    from pydantic import ConfigDict

    _HAS_CONFIG_DICT = True
except ImportError:
    _HAS_CONFIG_DICT = False


class Settings(BaseSettings):
    """Central settings object.  All values are readable from the
    environment, which makes twelve-factor configuration trivial."""

    if _HAS_CONFIG_DICT:
        model_config = ConfigDict(env_prefix="", case_sensitive=False, protected_namespaces=())

    app_name: str = "Price-Prophet"
    debug: bool = False
    version: str = "1.0.0"

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

    if not _HAS_CONFIG_DICT:

        class Config:
            env_prefix = ""
            case_sensitive = False


# Module-level singleton - import this everywhere instead of
# instantiating Settings directly.
settings = Settings()
