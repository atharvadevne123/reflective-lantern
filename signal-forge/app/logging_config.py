"""Structured logging configuration for Signal-Forge."""

from __future__ import annotations

import logging
import logging.config
import os
from typing import Any


def get_logging_config(log_level: str = "INFO") -> dict[str, Any]:
    """Return a dictConfig-compatible logging configuration."""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
            "json": {
                "()": "logging.Formatter",
                "fmt": '{"time":"%(asctime)s","name":"%(name)s","level":"%(levelname)s","msg":"%(message)s"}',
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": log_level.upper(),
            "handlers": ["console"],
        },
    }


def configure_logging(log_level: str | None = None) -> None:
    """Apply logging configuration from environment or supplied level."""
    level = log_level or os.environ.get("LOG_LEVEL", "INFO")
    logging.config.dictConfig(get_logging_config(level))
    logging.getLogger(__name__).info("Logging configured at level=%s", level)
