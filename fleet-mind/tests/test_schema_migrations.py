"""Tests for Fleet-Mind schema migration helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_fleet_schema.db")

import app.database as db_module
from app.database import Base


@pytest.fixture(scope="module")
def test_engine(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("db") / "schema_test.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    orig = db_module._engine
    db_module._engine = engine
    yield engine
    db_module._engine = orig
    engine.dispose()


def test_ensure_tables_exist_runs_without_error(test_engine):
    from app.schema_migrations import ensure_tables_exist
    ensure_tables_exist()


def test_column_exists_on_known_column(test_engine):
    from app.schema_migrations import column_exists
    assert column_exists("vehicle_predictions", "vehicle_id")


def test_column_exists_on_missing_column(test_engine):
    from app.schema_migrations import column_exists
    assert not column_exists("vehicle_predictions", "nonexistent_col_xyz")


def test_add_column_if_missing_adds_new_column(test_engine):
    from app.schema_migrations import add_column_if_missing, column_exists
    result = add_column_if_missing("vehicle_predictions", "test_new_col", "FLOAT DEFAULT 0")
    assert result is True
    assert column_exists("vehicle_predictions", "test_new_col")


def test_add_column_if_missing_skips_existing(test_engine):
    from app.schema_migrations import add_column_if_missing
    result = add_column_if_missing("vehicle_predictions", "vehicle_id", "TEXT")
    assert result is False
