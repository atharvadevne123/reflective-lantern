"""Tests for shared utility functions."""

from __future__ import annotations

import time

import pytest
from app.utils import (
    Timer,
    clamp,
    content_hash,
    generate_request_id,
    normalise_text,
    safe_divide,
)


class TestGenerateRequestId:
    def test_returns_string(self) -> None:
        assert isinstance(generate_request_id(), str)

    def test_ids_are_unique(self) -> None:
        ids = {generate_request_id() for _ in range(100)}
        assert len(ids) == 100

    def test_format_has_hyphens(self) -> None:
        rid = generate_request_id()
        assert rid.count("-") == 4


class TestNormaliseText:
    def test_lowercases(self) -> None:
        assert normalise_text("HELLO WORLD") == "hello world"

    def test_collapses_whitespace(self) -> None:
        assert normalise_text("hello   world") == "hello world"

    def test_strips_edges(self) -> None:
        assert normalise_text("  hi  ") == "hi"

    def test_empty_string(self) -> None:
        assert normalise_text("") == ""

    def test_cached(self) -> None:
        t1 = normalise_text("Test Text")
        t2 = normalise_text("Test Text")
        assert t1 == t2


class TestContentHash:
    def test_returns_64_chars(self) -> None:
        assert len(content_hash("hello")) == 64

    def test_same_text_same_hash(self) -> None:
        assert content_hash("foo bar") == content_hash("foo bar")

    def test_different_text_different_hash(self) -> None:
        assert content_hash("foo") != content_hash("bar")

    def test_hex_string(self) -> None:
        h = content_hash("test")
        int(h, 16)  # should not raise


class TestClamp:
    def test_below_lo(self) -> None:
        assert clamp(-1.0, 0.0, 1.0) == 0.0

    def test_above_hi(self) -> None:
        assert clamp(2.0, 0.0, 1.0) == 1.0

    def test_within_range(self) -> None:
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_at_boundaries(self) -> None:
        assert clamp(0.0, 0.0, 1.0) == 0.0
        assert clamp(1.0, 0.0, 1.0) == 1.0


class TestTimer:
    def test_elapsed_ms_positive(self) -> None:
        with Timer() as t:
            time.sleep(0.001)
        assert t.elapsed_ms > 0.0

    def test_elapsed_ms_reasonable(self) -> None:
        with Timer() as t:
            pass
        assert t.elapsed_ms < 100.0  # should be sub-100ms


class TestSafeDivide:
    def test_normal_division(self) -> None:
        assert safe_divide(10.0, 2.0) == pytest.approx(5.0)

    def test_zero_denominator_returns_default(self) -> None:
        assert safe_divide(10.0, 0.0) == 0.0

    def test_custom_default(self) -> None:
        assert safe_divide(1.0, 0.0, default=-1.0) == -1.0

    def test_zero_numerator(self) -> None:
        assert safe_divide(0.0, 5.0) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "value,lo,hi,expected",
    [
        (-5.0, 0.0, 10.0, 0.0),
        (15.0, 0.0, 10.0, 10.0),
        (5.0, 0.0, 10.0, 5.0),
        (0.0, 0.0, 1.0, 0.0),
    ],
)
def test_clamp_parametrized(value: float, lo: float, hi: float, expected: float) -> None:
    assert clamp(value, lo, hi) == expected
