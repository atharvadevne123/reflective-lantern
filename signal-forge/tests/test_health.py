"""Tests for health check utilities."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.health import check_database, check_model, full_health_report, get_uptime


def test_database_check_ok(db_session):
    result = check_database(db_session)
    assert result["status"] == "ok"


def test_get_uptime_positive():
    uptime = get_uptime()
    assert uptime >= 0.0


def test_model_check_returns_dict():
    result = check_model()
    assert "status" in result
    assert result["status"] in ("ok", "cold_start", "error")


def test_full_health_report_structure(db_session):
    report = full_health_report(db_session)
    assert report["status"] == "ok"
    assert "database" in report
    assert "model" in report
    assert "uptime_seconds" in report


@pytest.mark.parametrize("field", ["status", "database", "model", "uptime_seconds"])
def test_full_health_report_fields(db_session, field):
    report = full_health_report(db_session)
    assert field in report
