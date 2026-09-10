"""Structured logging configuration for Property-Sage.

Configures a JSON-formatted root logger suitable for log aggregation
platforms (Datadog, Loki, CloudWatch). Falls back to plain-text format
when JSON logging is disabled via the LOG_FORMAT environment variable.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any


class JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object.

    Attributes:
        extra_fields: Additional key-value pairs merged into every record.
    """

    def __init__(self, extra_fields: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """Serialise a log record to a JSON string.

        Args:
            record: The logging record to format.

        Returns:
            JSON-encoded log line.
        """
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
            **self.extra_fields,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(
    level: str | None = None,
    json_format: bool | None = None,
    service: str = "property-sage",
) -> None:
    """Apply structured logging configuration to the root logger.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR). Defaults to LOG_LEVEL env var.
        json_format: If True, use JSON formatter. Defaults to LOG_FORMAT==json env var.
        service: Service name injected into every JSON log record.
    """
    resolved_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    use_json = json_format if json_format is not None else os.getenv("LOG_FORMAT", "text") == "json"

    handler = logging.StreamHandler()
    if use_json:
        handler.setFormatter(JsonFormatter(extra_fields={"service": service}))
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))

    root = logging.getLogger()
    root.setLevel(resolved_level)
    root.handlers.clear()
    root.addHandler(handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
