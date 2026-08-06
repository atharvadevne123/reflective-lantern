"""Logging configuration for Reflective Lantern."""

from __future__ import annotations

import json
import logging
import traceback
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            data["exception"] = "".join(traceback.format_exception(*record.exc_info)).strip()
        run_id = getattr(record, "run_id", None)
        if run_id is not None:
            data["run_id"] = run_id
        return json.dumps(data)


def configure_logging(level: str = "INFO", *, json_logs: bool = False) -> None:
    """Configure the root logger with the given level and optional JSON formatter."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    root.handlers = [h for h in root.handlers if not isinstance(h, logging.StreamHandler)]
    if not root.handlers:
        pass

    handler = logging.StreamHandler()
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    root.addHandler(handler)

    for noisy in ("urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)


__all__ = ["JsonFormatter", "configure_logging", "get_logger"]
