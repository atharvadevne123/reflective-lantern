"""Extended tests for Property-Sage validators and sanitizers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.validators import (
    normalize_neighborhood,
    sanitize_request_id,
    validate_year_built_value,
)


def test_validate_year_built_valid():
    result = validate_year_built_value(2000)
    assert result == 2000


def test_validate_year_built_minimum_1800():
    result = validate_year_built_value(1800)
    assert result == 1800


def test_validate_year_built_too_old_raises():
    with pytest.raises(ValueError):
        validate_year_built_value(1799)


def test_validate_year_built_far_future_raises():
    with pytest.raises(ValueError):
        validate_year_built_value(3000)


def test_validate_year_built_recent():
    result = validate_year_built_value(2020)
    assert result == 2020


def test_normalize_neighborhood_lowercases():
    result = normalize_neighborhood("Downtown")
    assert result == "downtown"


def test_normalize_neighborhood_strips_whitespace():
    result = normalize_neighborhood("  midtown  ")
    assert result == "midtown"


def test_normalize_neighborhood_empty_returns_unknown():
    result = normalize_neighborhood("")
    assert result == "unknown"


def test_normalize_neighborhood_whitespace_only_returns_unknown():
    result = normalize_neighborhood("   ")
    assert result == "unknown"


def test_sanitize_request_id_removes_special_chars():
    result = sanitize_request_id("req!@#$id")
    assert "!" not in result
    assert "@" not in result
    assert "#" not in result


def test_sanitize_request_id_keeps_hyphens():
    result = sanitize_request_id("req-123-abc")
    assert result == "req-123-abc"


def test_sanitize_request_id_keeps_alphanumeric():
    result = sanitize_request_id("abc123")
    assert result == "abc123"


def test_sanitize_request_id_truncates_to_64():
    long_id = "a" * 100
    result = sanitize_request_id(long_id)
    assert len(result) <= 64


def test_sanitize_request_id_empty():
    result = sanitize_request_id("")
    assert result == ""


@pytest.mark.parametrize("year", [1850, 1900, 1950, 2000, 2024])
def test_validate_year_built_range(year):
    result = validate_year_built_value(year)
    assert result == year
