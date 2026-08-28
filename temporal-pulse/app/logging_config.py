"""Structured JSON logging configuration for Temporal-Pulse."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise *record* to a JSON string with standard fields."""
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = self.formatException(record.exc_info)
        for attr in ("correlation_id", "sensor_id", "endpoint"):
            value = getattr(record, attr, None)
            if value is not None:
                payload[attr] = value
        return json.dumps(payload, default=str)


def configure_logging(level: str | None = None, json_format: bool | None = None) -> None:
    """Configure root logging.

    Args:
        level: Log level name; defaults to LOG_LEVEL env var or INFO.
        json_format: Emit JSON lines when True; defaults to LOG_JSON env var.
    """
    level = level or os.getenv("LOG_LEVEL", "INFO")
    if json_format is None:
        json_format = os.getenv("LOG_JSON", "false").lower() == "true"

    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(handler)

    # Quieten noisy third-party loggers
    for noisy in ("uvicorn.access", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
