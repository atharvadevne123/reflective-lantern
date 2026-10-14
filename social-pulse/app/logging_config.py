"""Structured logging configuration for Social-Pulse."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any


LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()

LOG_FORMAT: str = (
    "%(asctime)s %(levelname)-8s %(name)s [%(filename)s:%(lineno)d] %(message)s"
)


def configure_logging(level: str | None = None) -> None:
    """Configure root logger with consistent format and level.

    Args:
        level: Override log level string (e.g. "DEBUG"). Falls back to the
               LOG_LEVEL env var, then "INFO".
    """
    effective_level = (level or LOG_LEVEL).upper()
    numeric_level = getattr(logging, effective_level, logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format=LOG_FORMAT,
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Silence noisy third-party loggers
    for noisy in ("uvicorn.access", "sqlalchemy.engine", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).debug(
        "Logging configured at level=%s", effective_level
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger using the package hierarchy.

    Args:
        name: Typically ``__name__`` from the calling module.

    Returns:
        A :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)


def log_dict(logger: logging.Logger, level: int, msg: str, data: dict[str, Any]) -> None:
    """Log a message with a structured dict payload.

    Args:
        logger: The logger to use.
        level: Logging level constant (e.g. logging.INFO).
        msg: Human-readable message prefix.
        data: Key-value pairs to append to the log line.
    """
    pairs = " ".join(f"{k}={v!r}" for k, v in data.items())
    logger.log(level, "%s %s", msg, pairs)
