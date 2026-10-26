"""Structured JSON logging configuration."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure structured JSON logging for the application.

    Args:
        level: Logging level string (default INFO).
    """
    fmt = (
        '{"time":"%(asctime)s","level":"%(levelname)s",'
        '"logger":"%(name)s","msg":%(message)s,"module":"%(module)s"}'
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
