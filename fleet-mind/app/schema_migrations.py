"""Programmatic schema migration helpers for Fleet-Mind database."""

from __future__ import annotations

import logging

from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from app.database import Base, _get_engine

logger = logging.getLogger(__name__)


def ensure_tables_exist() -> None:
    """Create all ORM-declared tables that do not yet exist in the database.

    Safe to call on every startup — uses CREATE TABLE IF NOT EXISTS semantics
    via SQLAlchemy's ``checkfirst=True``.
    """
    engine = _get_engine()
    Base.metadata.create_all(bind=engine, checkfirst=True)
    logger.info("Schema migration: all tables verified/created")


def column_exists(table_name: str, column_name: str) -> bool:
    """Return True if the column exists in the given table.

    Args:
        table_name: Name of the database table.
        column_name: Name of the column to check.

    Returns:
        True when the column is present, False otherwise.
    """
    engine = _get_engine()
    inspector = inspect(engine)
    columns = [c["name"] for c in inspector.get_columns(table_name)]
    return column_name in columns


def add_column_if_missing(
    table_name: str,
    column_name: str,
    column_definition: str,
) -> bool:
    """Add a column to a table if it does not already exist.

    Args:
        table_name: Name of the database table to alter.
        column_name: Name of the new column.
        column_definition: SQL column definition string (e.g. 'FLOAT DEFAULT 0').

    Returns:
        True if the column was added, False if it already existed.
    """
    if column_exists(table_name, column_name):
        logger.debug("Column %s.%s already exists — skipping", table_name, column_name)
        return False

    engine = _get_engine()
    ddl = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
    try:
        with engine.begin() as conn:
            conn.execute(text(ddl))
        logger.info("Added column %s.%s (%s)", table_name, column_name, column_definition)
        return True
    except Exception as exc:
        logger.error("Failed to add column %s.%s: %s", table_name, column_name, exc)
        return False
