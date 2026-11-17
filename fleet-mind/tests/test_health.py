"""Tests for Fleet-Mind health check utilities."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.health import build_health_payload, check_db


def _mock_session(success=True):
    session = MagicMock()
    if not success:
        session.execute.side_effect = Exception("DB connection failed")
    return session


def test_check_db_returns_true_on_success():
    db = _mock_session(success=True)
    result = check_db(db)
    assert result is True


def test_check_db_returns_false_on_exception():
    db = _mock_session(success=False)
    result = check_db(db)
    assert result is False


def test_build_health_payload_ok_when_model_and_db_ready():
    db = _mock_session(success=True)
    payload = build_health_payload(model_loaded=True, db=db)
    assert payload["status"] == "ok"


def test_build_health_payload_degraded_when_model_not_loaded():
    db = _mock_session(success=True)
    payload = build_health_payload(model_loaded=False, db=db)
    assert payload["status"] == "degraded"


def test_build_health_payload_degraded_when_db_fails():
    db = _mock_session(success=False)
    payload = build_health_payload(model_loaded=True, db=db)
    assert payload["status"] == "degraded"


def test_build_health_payload_has_required_keys():
    db = _mock_session(success=True)
    payload = build_health_payload(model_loaded=True, db=db)
    assert "status" in payload
    assert "model_version" in payload
    assert "db_ok" in payload


def test_build_health_payload_db_ok_bool():
    db = _mock_session(success=True)
    payload = build_health_payload(model_loaded=True, db=db)
    assert isinstance(payload["db_ok"], bool)


def test_build_health_payload_model_version_string():
    db = _mock_session(success=True)
    payload = build_health_payload(model_loaded=True, db=db)
    assert isinstance(payload["model_version"], str)
