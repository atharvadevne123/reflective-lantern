"""Tests for app.string_utils module."""

from __future__ import annotations

import pytest

from app.string_utils import (
    camel_to_snake,
    sanitize_identifier,
    slugify,
    snake_to_camel,
    truncate,
)


class TestTruncate:
    def test_short_string_unchanged(self) -> None:
        assert truncate("hello", 10) == "hello"

    def test_exact_length_unchanged(self) -> None:
        assert truncate("hello", 5) == "hello"

    def test_long_string_truncated(self) -> None:
        result = truncate("hello world", 8)
        assert len(result) == 8
        assert result.endswith("...")

    def test_custom_suffix(self) -> None:
        result = truncate("abcdefgh", 5, suffix="--")
        assert result == "abc--"

    def test_max_length_less_than_suffix_raises(self) -> None:
        with pytest.raises(ValueError):
            truncate("hello", 1, suffix="...")

    @pytest.mark.parametrize(
        ("text", "max_len", "expected_len"),
        [("a" * 50, 10, 10), ("short", 100, 5)],
    )
    def test_length_contract(self, text: str, max_len: int, expected_len: int) -> None:
        assert len(truncate(text, max_len)) == expected_len


class TestSlugify:
    def test_simple_string(self) -> None:
        assert slugify("Hello World") == "hello-world"

    def test_special_characters_removed(self) -> None:
        assert slugify("foo@bar!") == "foobar"

    def test_multiple_spaces_collapsed(self) -> None:
        assert slugify("a   b") == "a-b"

    def test_trailing_hyphens_stripped(self) -> None:
        result = slugify("  test  ")
        assert not result.startswith("-")
        assert not result.endswith("-")

    def test_unicode_normalised(self) -> None:
        result = slugify("café")
        assert result == "cafe"

    @pytest.mark.parametrize(
        ("input_text", "expected"),
        [
            ("My Title", "my-title"),
            ("hello-world", "hello-world"),
            ("123 abc", "123-abc"),
        ],
    )
    def test_parametrized(self, input_text: str, expected: str) -> None:
        assert slugify(input_text) == expected


class TestCamelToSnake:
    def test_simple(self) -> None:
        assert camel_to_snake("MyClass") == "my_class"

    def test_multiple_words(self) -> None:
        assert camel_to_snake("MyClassName") == "my_class_name"

    def test_already_snake(self) -> None:
        assert camel_to_snake("my_class") == "my_class"

    def test_all_caps_prefix(self) -> None:
        result = camel_to_snake("HTTPResponseCode")
        assert "http" in result

    @pytest.mark.parametrize(
        ("input_name", "expected"),
        [
            ("CamelCase", "camel_case"),
            ("camelCase", "camel_case"),
            ("simple", "simple"),
        ],
    )
    def test_parametrized(self, input_name: str, expected: str) -> None:
        assert camel_to_snake(input_name) == expected


class TestSnakeToCamel:
    def test_simple(self) -> None:
        assert snake_to_camel("my_var") == "myVar"

    def test_multiple_parts(self) -> None:
        assert snake_to_camel("my_variable_name") == "myVariableName"

    def test_single_word_unchanged(self) -> None:
        assert snake_to_camel("hello") == "hello"

    def test_leading_underscore(self) -> None:
        result = snake_to_camel("_private")
        assert isinstance(result, str)

    @pytest.mark.parametrize(
        ("snake", "camel"),
        [
            ("first_name", "firstName"),
            ("last_name", "lastName"),
            ("get_by_id", "getById"),
        ],
    )
    def test_parametrized(self, snake: str, camel: str) -> None:
        assert snake_to_camel(snake) == camel


class TestSanitizeIdentifier:
    def test_clean_string_unchanged(self) -> None:
        assert sanitize_identifier("valid_name") == "valid_name"

    def test_spaces_replaced(self) -> None:
        assert sanitize_identifier("my name") == "my_name"

    def test_special_chars_replaced(self) -> None:
        assert sanitize_identifier("a-b.c") == "a_b_c"

    def test_leading_digit_prefixed(self) -> None:
        result = sanitize_identifier("123abc")
        assert result.startswith("_")

    def test_empty_string_returns_underscore(self) -> None:
        assert sanitize_identifier("") == "_"

    @pytest.mark.parametrize(
        ("input_name", "expected"),
        [
            ("hello", "hello"),
            ("hello world", "hello_world"),
            ("9times", "_9times"),
        ],
    )
    def test_parametrized(self, input_name: str, expected: str) -> None:
        assert sanitize_identifier(input_name) == expected
