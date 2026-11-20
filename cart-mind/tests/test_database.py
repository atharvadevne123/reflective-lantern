"""Tests for cart-mind app/database.py SQLAlchemy models."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database import Base, DriftLog, ItemCatalog, PredictionLog, UserProfile


@pytest.fixture(scope="module")
def engine():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def session(engine):
    with Session(engine) as s:
        yield s


class TestPredictionLog:
    def test_create_record(self, session: Session) -> None:
        record = PredictionLog(
            user_id="u1",
            item_id="i1",
            intent_score=0.75,
        )
        session.add(record)
        session.commit()
        assert record.id is not None

    def test_defaults(self, session: Session) -> None:
        record = PredictionLog(user_id="u2", item_id="i2", intent_score=0.5)
        session.add(record)
        session.commit()
        assert record.created_at is not None


class TestDriftLog:
    def test_create_record(self, session: Session) -> None:
        record = DriftLog(feature="price", drift_score=0.12)
        session.add(record)
        session.commit()
        assert record.id is not None


class TestUserProfile:
    def test_create_record(self, session: Session) -> None:
        profile = UserProfile(user_id="user_abc")
        session.add(profile)
        session.commit()
        assert profile.user_id == "user_abc"


class TestItemCatalog:
    def test_create_record(self, session: Session) -> None:
        item = ItemCatalog(item_id="item_xyz", category="electronics")
        session.add(item)
        session.commit()
        assert item.item_id == "item_xyz"
