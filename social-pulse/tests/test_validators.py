"""Tests for input validation helpers."""

from __future__ import annotations

import pytest

from app.validators import (
    check_repetitive_content,
    sanitise_text,
    validate_hashtag_consistency,
    validate_mention_consistency,
    validate_sample,
)


class TestSanitiseText:
    def test_strips_whitespace(self) -> None:
        assert sanitise_text("  hello  ") == "hello"

    def test_removes_null_bytes(self) -> None:
        result = sanitise_text("hello\x00world")
        assert "\x00" not in result

    def test_preserves_emojis(self) -> None:
        result = sanitise_text("hello 😀")
        assert "😀" in result

    def test_raises_on_blank(self) -> None:
        with pytest.raises(ValueError):
            sanitise_text("   ")

    def test_normalises_unicode(self) -> None:
        result = sanitise_text("café")  # e + combining accent
        assert len(result) > 0


class TestValidateHashtagConsistency:
    def test_counts_hashtags(self) -> None:
        assert validate_hashtag_consistency("#a #b #c", 0) == 3

    def test_ignores_client_count(self) -> None:
        assert validate_hashtag_consistency("#tag", 99) == 1

    def test_no_hashtags(self) -> None:
        assert validate_hashtag_consistency("no tags", 5) == 0


class TestValidateMentionConsistency:
    def test_counts_mentions(self) -> None:
        assert validate_mention_consistency("@alice @bob", 0) == 2

    def test_no_mentions(self) -> None:
        assert validate_mention_consistency("no mentions", 3) == 0


class TestCheckRepetitiveContent:
    def test_detects_long_run(self) -> None:
        assert check_repetitive_content("a" * 35) is True

    def test_normal_text_passes(self) -> None:
        assert check_repetitive_content("Hello world! #trending") is False

    def test_exactly_at_threshold_not_flagged(self) -> None:
        assert check_repetitive_content("a" * 29) is False

    def test_one_over_threshold_flagged(self) -> None:
        assert check_repetitive_content("a" * 30) is True


class TestValidateSample:
    def _base_sample(self) -> dict:
        return {
            "text": "Hello #world @user",
            "author_followers": 1000,
            "has_media": False,
            "has_link": False,
            "hashtag_count": 5,  # will be corrected to 1
            "mention_count": 3,  # will be corrected to 1
            "hour_of_day": 10,
            "day_of_week": 2,
        }

    def test_corrects_hashtag_count(self) -> None:
        s = validate_sample(self._base_sample())
        assert s["hashtag_count"] == 1

    def test_corrects_mention_count(self) -> None:
        s = validate_sample(self._base_sample())
        assert s["mention_count"] == 1

    def test_clamps_negative_followers(self) -> None:
        base = self._base_sample()
        base["author_followers"] = -100
        s = validate_sample(base)
        assert s["author_followers"] == 0

    def test_clamps_hour_above_23(self) -> None:
        base = self._base_sample()
        base["hour_of_day"] = 25
        s = validate_sample(base)
        assert s["hour_of_day"] == 23

    def test_clamps_dow_above_6(self) -> None:
        base = self._base_sample()
        base["day_of_week"] = 8
        s = validate_sample(base)
        assert s["day_of_week"] == 6


@pytest.mark.parametrize(
    "text,expected",
    [
        ("#a", 1),
        ("#a #b", 2),
        ("#a #b #c #d", 4),
        ("no tags", 0),
    ],
)
def test_hashtag_count_parametrized(text: str, expected: int) -> None:
    assert validate_hashtag_consistency(text, 0) == expected
