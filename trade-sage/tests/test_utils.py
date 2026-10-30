"""Tests for general-purpose utility functions."""

from __future__ import annotations

import pytest

from app.utils import clamp, flatten_dict, hash_payload, percentage_change, safe_round


class TestSafeRound:
    def test_rounds_float(self) -> None:
        assert safe_round(3.14159, 2) == 3.14

    def test_none_returns_none(self) -> None:
        assert safe_round(None) is None

    def test_default_ndigits_four(self) -> None:
        result = safe_round(1.23456789)
        assert result == 1.2346

    @pytest.mark.parametrize("value,ndigits,expected", [
        (0.5555, 2, 0.56),
        (1.0, 0, 1.0),
        (-3.7, 0, -4.0),
    ])
    def test_parametrized_rounding(self, value: float, ndigits: int, expected: float) -> None:
        assert safe_round(value, ndigits) == expected


class TestFlattenDict:
    def test_flat_dict_unchanged(self) -> None:
        d = {"a": 1, "b": 2}
        assert flatten_dict(d) == d

    def test_nested_dict_flattened(self) -> None:
        d = {"a": {"b": 1, "c": 2}, "d": 3}
        result = flatten_dict(d)
        assert result == {"a.b": 1, "a.c": 2, "d": 3}

    def test_deeply_nested(self) -> None:
        d = {"x": {"y": {"z": 42}}}
        result = flatten_dict(d)
        assert result == {"x.y.z": 42}

    def test_empty_dict_returns_empty(self) -> None:
        assert flatten_dict({}) == {}

    def test_custom_separator(self) -> None:
        d = {"a": {"b": 1}}
        result = flatten_dict(d, sep="/")
        assert "a/b" in result


class TestHashPayload:
    def test_returns_string(self) -> None:
        assert isinstance(hash_payload("test"), str)

    def test_length_is_sixteen(self) -> None:
        assert len(hash_payload("hello world")) == 16

    def test_same_input_same_hash(self) -> None:
        assert hash_payload("abc") == hash_payload("abc")

    def test_different_input_different_hash(self) -> None:
        assert hash_payload("abc") != hash_payload("xyz")

    def test_empty_string(self) -> None:
        result = hash_payload("")
        assert len(result) == 16


class TestClamp:
    def test_value_in_range_unchanged(self) -> None:
        assert clamp(5.0, 0.0, 10.0) == 5.0

    def test_value_below_lo_clipped(self) -> None:
        assert clamp(-5.0, 0.0, 10.0) == 0.0

    def test_value_above_hi_clipped(self) -> None:
        assert clamp(15.0, 0.0, 10.0) == 10.0

    def test_value_equal_lo(self) -> None:
        assert clamp(0.0, 0.0, 1.0) == 0.0

    def test_value_equal_hi(self) -> None:
        assert clamp(1.0, 0.0, 1.0) == 1.0

    @pytest.mark.parametrize("v,lo,hi,expected", [
        (0.5, 0, 1, 0.5),
        (-10, -5, 5, -5),
        (100, 0, 50, 50),
    ])
    def test_parametrized(self, v, lo, hi, expected) -> None:
        assert clamp(v, lo, hi) == expected


class TestPercentageChange:
    def test_positive_change(self) -> None:
        result = percentage_change(100.0, 110.0)
        assert abs(result - 10.0) < 1e-3

    def test_negative_change(self) -> None:
        result = percentage_change(100.0, 90.0)
        assert abs(result - (-10.0)) < 1e-3

    def test_zero_old_returns_none(self) -> None:
        assert percentage_change(0.0, 5.0) is None

    def test_no_change_returns_zero(self) -> None:
        result = percentage_change(100.0, 100.0)
        assert result == 0.0
