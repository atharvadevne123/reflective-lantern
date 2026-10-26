"""Typed configuration schema helpers for environment-driven settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ConfigSchema",
    "load_from_env",
    "env_bool",
    "env_int",
    "env_float",
]


def env_bool(key: str, default: bool = False) -> bool:
    """Read *key* from environment and convert to bool.

    Truthy strings: ``"1"``, ``"true"``, ``"yes"``, ``"on"`` (case-insensitive).

    Args:
        key: Environment variable name.
        default: Returned when the variable is absent.
    """
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(key: str, default: int = 0) -> int:
    """Read *key* from environment and parse as integer.

    Args:
        key: Environment variable name.
        default: Returned when the variable is absent.

    Raises:
        ValueError: If the value cannot be parsed as int.
    """
    raw = os.environ.get(key)
    if raw is None:
        return default
    return int(raw.strip())


def env_float(key: str, default: float = 0.0) -> float:
    """Read *key* from environment and parse as float.

    Args:
        key: Environment variable name.
        default: Returned when the variable is absent.

    Raises:
        ValueError: If the value cannot be parsed as float.
    """
    raw = os.environ.get(key)
    if raw is None:
        return default
    return float(raw.strip())


@dataclass
class ConfigSchema:
    """Container for application configuration loaded from environment variables.

    Attributes:
        debug: Enable debug mode.
        log_level: Logging level string.
        max_workers: Thread-pool or worker count.
        request_timeout: HTTP request timeout in seconds.
        extra: Arbitrary additional settings.
    """

    debug: bool = False
    log_level: str = "INFO"
    max_workers: int = 4
    request_timeout: float = 30.0
    extra: dict[str, Any] = field(default_factory=dict)

    def is_production(self) -> bool:
        """Return True when debug is off (proxy for production mode)."""
        return not self.debug

    def as_dict(self) -> dict[str, Any]:
        """Return a plain dict representation of the configuration."""
        return {
            "debug": self.debug,
            "log_level": self.log_level,
            "max_workers": self.max_workers,
            "request_timeout": self.request_timeout,
            "extra": self.extra,
        }


def load_from_env(prefix: str = "APP_") -> ConfigSchema:
    """Build a :class:`ConfigSchema` from environment variables with *prefix*.

    Reads ``{prefix}DEBUG``, ``{prefix}LOG_LEVEL``, ``{prefix}MAX_WORKERS``,
    and ``{prefix}REQUEST_TIMEOUT``.

    Args:
        prefix: Namespace prefix for environment variable names.

    Returns:
        Populated :class:`ConfigSchema`.
    """
    return ConfigSchema(
        debug=env_bool(f"{prefix}DEBUG", default=False),
        log_level=os.environ.get(f"{prefix}LOG_LEVEL", "INFO"),
        max_workers=env_int(f"{prefix}MAX_WORKERS", default=4),
        request_timeout=env_float(f"{prefix}REQUEST_TIMEOUT", default=30.0),
    )
