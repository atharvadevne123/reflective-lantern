"""Tests for Sports-Oracle SQLAlchemy ORM models and session management."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.database as db_module
from app.database import Base, Match, ModelMetrics, create_tables, get_engine, get_session


@pytest.fixture(autouse=True)
def reset_engine():
    db_module._engine = None
    with patch.dict(os.environ, {"DATABASE_URL": "sqlite:///:memory:"}):
        yield
    db_module._engine = None


def test_get_engine_returns_engine():
    engine = get_engine()
    assert engine is not None


def test_get_engine_sqlite_in_memory():
    engine = get_engine()
    assert "sqlite" in str(engine.url)


def test_get_engine_singleton():
    e1 = get_engine()
    e2 = get_engine()
    assert e1 is e2


def test_create_tables_does_not_raise():
    create_tables()


def test_get_session_returns_session():
    create_tables()
    session = get_session()
    assert session is not None
    session.close()


def test_match_model_has_required_columns():
    cols = {c.name for c in Match.__table__.columns}
    assert "id" in cols
    assert "match_id" in cols
    assert "home_team" in cols
    assert "away_team" in cols
    assert "outcome" in cols


def test_match_model_tablename():
    assert Match.__tablename__ == "matches"


def test_model_metrics_tablename():
    assert ModelMetrics.__tablename__ == "model_metrics"


def test_model_metrics_has_required_columns():
    cols = {c.name for c in ModelMetrics.__table__.columns}
    assert "id" in cols
    assert "run_id" in cols
    assert "auc_mean" in cols


def test_insert_and_query_match():
    create_tables()
    session = get_session()
    try:
        m = Match(
            match_id="m001",
            home_team="Team A",
            away_team="Team B",
            outcome="H",
        )
        session.add(m)
        session.commit()
        result = session.query(Match).filter_by(match_id="m001").first()
        assert result is not None
        assert result.home_team == "Team A"
    finally:
        session.close()
