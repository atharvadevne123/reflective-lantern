"""Structured logging configuration supporting plain and JSON formats."""

from __future__ import annotations

import logging
import sys


class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        import traceback

        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = traceback.format_exception(*record.exc_info)
        if hasattr(record, "correlation_id"):
            payload["correlation_id"] = record.correlation_id
        return json.dumps(payload)


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure root logger once at application startup."""
    handler = logging.StreamHandler(sys.stdout)
    if json_output:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root.handlers:
        root.addHandler(handler)
    else:
        root.handlers[0] = handler
