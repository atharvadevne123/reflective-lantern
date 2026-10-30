"""Tests for sanitize_request_id."""
from app.validators import sanitize_request_id


def test_alphanumeric_unchanged():
    assert sanitize_request_id("abc-123") == "abc-123"


def test_strips_special_chars():
    assert sanitize_request_id("req_!@#id") == "reqid"


def test_max_length():
    long_id = "a" * 100
    assert len(sanitize_request_id(long_id)) == 64


def test_empty():
    assert sanitize_request_id("") == ""


def test_keeps_hyphens():
    assert sanitize_request_id("req-uuid-1234") == "req-uuid-1234"
