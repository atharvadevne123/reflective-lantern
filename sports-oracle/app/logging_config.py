"""Structured JSON logging configuration for Sports-Oracle."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure structured JSON logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
        stream=sys.stdout,
        force=True,
    )
    # Suppress noisy library loggers
    for noisy in ("uvicorn.access", "sqlalchemy.engine", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
