"""Tests for veritas-rag text normalization."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.normalize import normalize_text, strip_email_quotes, truncate_to_bytes


def test_normalize_strips_leading_trailing_whitespace():
    assert normalize_text("  hello  ") == "hello"


def test_normalize_collapses_multiple_spaces():
    result = normalize_text("hello   world")
    assert "   " not in result
    assert result == "hello world"


def test_normalize_removes_control_chars():
    result = normalize_text("hello\x00world\x07")
    assert "\x00" not in result
    assert "\x07" not in result


def test_normalize_unifies_crlf():
    result = normalize_text("line1\r\nline2")
    assert "\r" not in result
    assert "line1" in result and "line2" in result


def test_normalize_collapses_many_newlines():
    result = normalize_text("a\n\n\n\n\nb")
    assert "\n\n\n" not in result


def test_normalize_unicode_nfkc():
    # NFKC: ligature fi -> fi
    result = normalize_text("ﬁle")
    assert result == "file"


def test_normalize_empty_string():
    assert normalize_text("") == ""


def test_strip_email_quotes_removes_prefix():
    result = strip_email_quotes("> quoted line")
    assert not result.strip().startswith(">")


def test_strip_email_quotes_multi_level():
    result = strip_email_quotes(">> double quoted")
    assert ">>" not in result


def test_strip_email_quotes_preserves_non_quoted():
    text = "normal line\n> quoted\nnormal again"
    result = strip_email_quotes(text)
    assert "normal line" in result


def test_truncate_to_bytes_within_limit():
    text = "hello world"
    result = truncate_to_bytes(text, max_bytes=1000)
    assert result == text


def test_truncate_to_bytes_truncates():
    text = "a" * 100
    result = truncate_to_bytes(text, max_bytes=10)
    assert len(result.encode("utf-8")) <= 10


def test_truncate_to_bytes_valid_utf8():
    text = "café " * 50
    result = truncate_to_bytes(text, max_bytes=20)
    result.encode("utf-8")  # should not raise


@pytest.mark.parametrize(
    "text,expected_len",
    [
        ("", 0),
        ("abc", 3),
    ],
)
def test_normalize_known_outputs(text, expected_len):
    result = normalize_text(text)
    assert len(result) == expected_len


def test_normalize_tabs_to_space():
    result = normalize_text("hello\tworld")
    assert "\t" not in result
    assert result == "hello world"


def test_normalize_form_feed_removed():
    result = normalize_text("page1\fpage2")
    assert "\f" not in result


def test_normalize_preserves_newlines_within_limit():
    result = normalize_text("line1\nline2")
    assert "line1" in result
    assert "line2" in result


def test_strip_email_quotes_nested_whitespace():
    result = strip_email_quotes(">> deeply quoted")
    assert not result.strip().startswith(">")


def test_truncate_to_bytes_exact_boundary():
    text = "a" * 10
    result = truncate_to_bytes(text, max_bytes=10)
    assert result == text
    assert len(result.encode("utf-8")) == 10


def test_truncate_to_bytes_zero():
    result = truncate_to_bytes("hello", max_bytes=0)
    assert result == ""


@pytest.mark.parametrize("max_bytes", [1, 5, 50, 500])
def test_truncate_to_bytes_always_valid_utf8(max_bytes):
    text = "日本語テスト" * 20
    result = truncate_to_bytes(text, max_bytes=max_bytes)
    assert len(result.encode("utf-8")) <= max_bytes
    result.encode("utf-8")  # must not raise


def test_normalize_multiple_blank_lines_collapsed():
    text = "line1\n\n\n\nline2"
    result = normalize_text(text)
    assert "\n\n\n" not in result


def test_normalize_leading_trailing_whitespace_stripped():
    result = normalize_text("  hello world  ")
    assert result == "hello world"


def test_normalize_control_characters_removed():
    text = "hello\x01\x02world"
    result = normalize_text(text)
    assert "\x01" not in result
    assert "\x02" not in result


def test_strip_email_quotes_multiple_lines():
    text = "> line one\n> line two\nreply"
    result = strip_email_quotes(text)
    assert ">" not in result.split("\n")[0]


def test_truncate_to_bytes_no_change_when_under_limit():
    text = "short text"
    result = truncate_to_bytes(text, max_bytes=1000)
    assert result == text


@pytest.mark.parametrize("text", [
    "hello world",
    "HELLO WORLD",
    "  spaces  ",
    "line1\nline2",
])
def test_normalize_text_is_idempotent(text):
    once = normalize_text(text)
    twice = normalize_text(once)
    assert once == twice
