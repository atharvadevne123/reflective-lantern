"""Tests for app.utils helper functions."""

import pytest

from app.utils import clamp, format_currency, is_finite_positive, percentage_change, slugify


def test_format_currency_basic():
    assert format_currency(1234.56) == "$1,234.56"


def test_format_currency_large():
    assert format_currency(1_000_000.0) == "$1,000,000.00"


def test_format_currency_negative():
    assert format_currency(-500.0) == "-$500.00"


def test_clamp_within():
    assert clamp(5.0, 0.0, 10.0) == 5.0


def test_clamp_below():
    assert clamp(-1.0, 0.0, 10.0) == 0.0


def test_clamp_above():
    assert clamp(15.0, 0.0, 10.0) == 10.0


def test_slugify_basic():
    assert slugify("Hello World") == "hello-world"


def test_slugify_special_chars():
    assert slugify("Price: $500") == "price-500"


def test_percentage_change_increase():
    assert percentage_change(100.0, 110.0) == 10.0


def test_percentage_change_zero_old():
    assert percentage_change(0.0, 100.0) == 0.0


@pytest.mark.parametrize("val,expected", [(1.0, True), (-1.0, False), (0.0, False)])
def test_is_finite_positive(val, expected):
    assert is_finite_positive(val) == expected
