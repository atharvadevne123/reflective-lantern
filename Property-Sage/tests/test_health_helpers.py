"""Tests for is_healthy and get_unhealthy_components."""

import pytest

from app.health import get_unhealthy_components, is_healthy

HEALTHY_REPORT = {"status": "ok", "components": {"db": "ok", "model": "ok"}}
DEGRADED_REPORT = {"status": "degraded", "components": {"db": "ok", "model": "missing"}}


def test_is_healthy_ok():
    assert is_healthy(HEALTHY_REPORT) is True


def test_is_healthy_degraded():
    assert is_healthy(DEGRADED_REPORT) is False


def test_get_unhealthy_empty():
    assert get_unhealthy_components(HEALTHY_REPORT) == []


def test_get_unhealthy_one():
    result = get_unhealthy_components(DEGRADED_REPORT)
    assert "model" in result


@pytest.mark.parametrize("status,expected", [("ok", True), ("degraded", False), ("error", False)])
def test_is_healthy_parametrized(status, expected):
    assert is_healthy({"status": status}) == expected
