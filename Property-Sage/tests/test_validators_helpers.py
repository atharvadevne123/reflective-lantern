"""Tests for sanitize_request_id and normalize_neighborhood."""

import pytest

from app.validators import normalize_neighborhood, sanitize_request_id


def test_sanitize_clean_id():
    assert sanitize_request_id("abc-123") == "abc-123"


def test_sanitize_special_chars_replaced():
    result = sanitize_request_id("abc!@#xyz")
    assert "!" not in result


def test_sanitize_uuid_preserved():
    uid = "550e8400-e29b-41d4-a716-446655440000"
    assert sanitize_request_id(uid) == uid


@pytest.mark.parametrize("raw,expected", [
    ("clean_id", "clean_id"),
    ("with space", "with-space"),
])
def test_sanitize_parametrized(raw, expected):
    assert sanitize_request_id(raw) == expected


def test_normalize_basic():
    assert normalize_neighborhood("Suburb") == "suburb"


def test_normalize_empty_returns_unknown():
    assert normalize_neighborhood("") == "unknown"


def test_normalize_whitespace_stripped():
    assert normalize_neighborhood("  downtown  ") == "downtown"


@pytest.mark.parametrize("raw,expected", [
    ("DOWNTOWN", "downtown"),
    ("SUBURB", "suburb"),
])
def test_normalize_case_insensitive(raw, expected):
    assert normalize_neighborhood(raw) == expected
