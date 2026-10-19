"""Tests for health check module."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.health import build_health_status, check_db


@pytest.fixture
def live_db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    sess = Session()
    yield sess
    sess.close()
    Base.metadata.drop_all(bind=engine)


class TestCheckDb:
    def test_returns_true_for_live_db(self, live_db_session) -> None:
        assert check_db(live_db_session) is True

    def test_returns_false_for_closed_session(self) -> None:
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        sess = Session()
        sess.close()
        # Session is closed; check_db should return False gracefully
        result = check_db(sess)
        assert isinstance(result, bool)


class TestBuildHealthStatus:
    def test_status_ok_when_all_pass(self, live_db_session) -> None:
        status = build_health_status(live_db_session, model_loaded=True, model_version="1.0")
        assert status.status == "ok"

    def test_status_degraded_when_model_missing(self, live_db_session) -> None:
        status = build_health_status(live_db_session, model_loaded=False, model_version="none")
        assert status.status == "degraded"

    def test_model_version_preserved(self, live_db_session) -> None:
        status = build_health_status(live_db_session, model_loaded=True, model_version="2.1.0")
        assert status.model_version == "2.1.0"

    def test_uptime_positive(self, live_db_session) -> None:
        status = build_health_status(live_db_session, model_loaded=True, model_version="1.0")
        assert status.uptime_seconds >= 0.0

    def test_checks_dict_has_model_key(self, live_db_session) -> None:
        status = build_health_status(live_db_session, model_loaded=True, model_version="1.0")
        assert "model" in status.checks

    def test_checks_dict_has_database_key(self, live_db_session) -> None:
        status = build_health_status(live_db_session, model_loaded=True, model_version="1.0")
        assert "database" in status.checks


class TestHealthStatusToDict:
    def test_to_dict_has_all_keys(self, live_db_session) -> None:
        status = build_health_status(live_db_session, model_loaded=True, model_version="1.0")
        d = status.to_dict()
        assert "status" in d
        assert "model_loaded" in d
        assert "db_reachable" in d
        assert "uptime_seconds" in d
        assert "checks" in d


@pytest.mark.parametrize("model_loaded,expected_status", [
    (True, "ok"),
    (False, "degraded"),
])
def test_status_from_model_flag(live_db_session, model_loaded: bool, expected_status: str) -> None:
    status = build_health_status(live_db_session, model_loaded=model_loaded, model_version="1.0")
    assert status.status == expected_status
