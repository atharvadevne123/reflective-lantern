"""Structured JSON logging configuration for Ticket-Oracle.

Configures the root logger to emit JSON lines in production and
human-readable format in development, controlled by LOG_FORMAT env var.
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import datetime, timezone


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        doc: dict = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge structured extras (skip internal logging attrs)
        skip = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "message",
        }
        for key, value in record.__dict__.items():
            if key not in skip:
                doc[key] = value

        if record.exc_info:
            doc["exc"] = traceback.format_exception(*record.exc_info)

        return json.dumps(doc, default=str)


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Configure the root logger.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        fmt: 'json' for JSON lines, anything else for human-readable.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if root.handlers:
        root.handlers.clear()

    handler = logging.StreamHandler()
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s %(message)s")
        )

    root.addHandler(handler)


def setup_from_env() -> None:
    """Read LOG_LEVEL and LOG_FORMAT from environment and call configure_logging."""
    level = os.getenv("LOG_LEVEL", "INFO")
    fmt = os.getenv("LOG_FORMAT", "json")
    configure_logging(level=level, fmt=fmt)
