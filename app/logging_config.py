"""Structured JSON logging configuration for Watt-Guard."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any


class JsonFormatter(logging.Formatter):
    """Emit log records as JSON lines for structured log ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise a log record to a single JSON line.

        Args:
            record: The log record to format.

        Returns:
            JSON-encoded string with ts, level, logger, msg and optional fields.
        """
        log: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            log["exc"] = self.formatException(record.exc_info)
        if hasattr(record, "request_id"):
            log["request_id"] = record.request_id
        if hasattr(record, "correlation_id"):
            log["correlation_id"] = record.correlation_id
        return json.dumps(log, ensure_ascii=False)


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Set up root logger with optional JSON formatting.

    Args:
        level: Logging level name (e.g. "INFO", "DEBUG").
        json_output: Emit structured JSON lines when True; plain text otherwise.
    """
    handler = logging.StreamHandler(sys.stdout)
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


class TraceIdFilter(logging.Filter):
    """Inject a trace_id field into every log record that lacks one."""

    def __init__(self, trace_id: str = "") -> None:
        super().__init__()
        self.trace_id = trace_id

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "trace_id"):
            record.trace_id = self.trace_id  # type: ignore[attr-defined]
        return True


def add_trace_id_filter(logger: logging.Logger, trace_id: str = "") -> None:
    """Attach a TraceIdFilter to *logger* so every record carries a trace_id.

    Args:
        logger: The logger instance to configure.
        trace_id: The trace identifier to stamp on records (e.g. a request UUID).
    """
    for handler in logger.handlers:
        handler.addFilter(TraceIdFilter(trace_id=trace_id))
    if not logger.handlers:
        logger.addFilter(TraceIdFilter(trace_id=trace_id))
