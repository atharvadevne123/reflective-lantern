"""Extended validator tests for social-pulse with parametrize and boundary coverage."""
from __future__ import annotations

import pytest

from app.validators import (
    check_repetitive_content,
    estimate_reading_time_seconds,
    is_alpha_text,
    sanitise_text,
    validate_hashtag_consistency,
    validate_mention_consistency,
    validate_sample,
    validate_url,
    word_count,
)


def test_sanitise_text_basic() -> None:
    assert sanitise_text("Hello World") == "Hello World"


def test_sanitise_text_strips_whitespace() -> None:
    assert sanitise_text("  hello  ") == "hello"


def test_sanitise_text_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        sanitise_text("")


def test_sanitise_text_whitespace_only_raises() -> None:
    with pytest.raises(ValueError):
        sanitise_text("   ")


def test_sanitise_text_control_chars_removed() -> None:
    result = sanitise_text("hello\x00world")
    assert "\x00" not in result


@pytest.mark.parametrize("text,expected", [
    ("#hello world", 1),
    ("#hello #world #foo", 3),
    ("no hashtags here", 0),
    ("#a #b", 2),
    ("text with no tags", 0),
])
def test_validate_hashtag_consistency(text: str, expected: int) -> None:
    result = validate_hashtag_consistency(text, 99)
    assert result == expected


@pytest.mark.parametrize("text,expected", [
    ("@alice hello", 1),
    ("@alice @bob @carol", 3),
    ("no mentions", 0),
    ("@a and @b", 2),
])
def test_validate_mention_consistency(text: str, expected: int) -> None:
    result = validate_mention_consistency(text, 99)
    assert result == expected


@pytest.mark.parametrize("text,threshold,expected", [
    ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 30, True),
    ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 30, True),
    ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 30, False),
    ("normal text here", 30, False),
    ("bbbbb", 5, True),
    ("bbbbb", 6, False),
])
def test_check_repetitive_content(text: str, threshold: int, expected: bool) -> None:
    assert check_repetitive_content(text, threshold=threshold) is expected


def test_check_repetitive_content_empty_string() -> None:
    assert check_repetitive_content("") is False


@pytest.mark.parametrize("text,has_url", [
    ("Check https://example.com for more", True),
    ("Visit http://foo.bar/path", True),
    ("No URLs here at all", False),
    ("ftp://not-matched.com", False),
    ("https://x.io", True),
])
def test_validate_url(text: str, has_url: bool) -> None:
    assert validate_url(text) is has_url


@pytest.mark.parametrize("text,wpm,expected_min", [
    ("one", 238, 1.0),
    ("word " * 238, 238, 60.0),
    ("   ", 238, 1.0),
])
def test_estimate_reading_time_minimum_one_second(text: str, wpm: int, expected_min: float) -> None:
    result = estimate_reading_time_seconds(text, wpm=wpm)
    assert result >= expected_min


def test_estimate_reading_time_proportional() -> None:
    short = estimate_reading_time_seconds("word " * 10, wpm=238)
    long = estimate_reading_time_seconds("word " * 100, wpm=238)
    assert long > short


@pytest.mark.parametrize("text,expected", [
    ("hello", True),
    ("123", False),
    ("", False),
    ("!@#$%", False),
    ("hello1", True),
])
def test_is_alpha_text(text: str, expected: bool) -> None:
    assert is_alpha_text(text) is expected


@pytest.mark.parametrize("text,expected", [
    ("one two three", 3),
    ("", 0),
    ("   ", 0),
    ("single", 1),
    ("a b c d e", 5),
])
def test_word_count(text: str, expected: int) -> None:
    assert word_count(text) == expected


def test_validate_sample_corrects_hashtag_count() -> None:
    sample = {
        "text": "Hello #world #python",
        "hashtag_count": 99,
        "mention_count": 0,
        "author_followers": 100,
        "hour_of_day": 12,
        "day_of_week": 1,
    }
    result = validate_sample(sample)
    assert result["hashtag_count"] == 2


def test_validate_sample_clips_hour_of_day() -> None:
    sample = {
        "text": "Hello",
        "hashtag_count": 0,
        "mention_count": 0,
        "author_followers": 100,
        "hour_of_day": 30,
        "day_of_week": 1,
    }
    result = validate_sample(sample)
    assert result["hour_of_day"] == 23


def test_validate_sample_clips_day_of_week() -> None:
    sample = {
        "text": "Hello",
        "hashtag_count": 0,
        "mention_count": 0,
        "author_followers": 100,
        "hour_of_day": 12,
        "day_of_week": 10,
    }
    result = validate_sample(sample)
    assert result["day_of_week"] == 6


def test_validate_sample_clips_negative_followers() -> None:
    sample = {
        "text": "Hello",
        "hashtag_count": 0,
        "mention_count": 0,
        "author_followers": -500,
        "hour_of_day": 12,
        "day_of_week": 1,
    }
    result = validate_sample(sample)
    assert result["author_followers"] == 0
