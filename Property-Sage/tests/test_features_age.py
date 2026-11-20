"""Tests for property_age_category helper."""

import pytest

from app.features import property_age_category


def test_new():
    assert property_age_category(2020, 2024) == "new"


def test_modern():
    assert property_age_category(2010, 2024) == "modern"


def test_established():
    assert property_age_category(1995, 2024) == "established"


def test_vintage():
    assert property_age_category(1970, 2024) == "vintage"


def test_historic():
    assert property_age_category(1900, 2024) == "historic"


@pytest.mark.parametrize("year,expected", [
    (2023, "new"), (2014, "modern"), (1999, "established"),
    (1964, "vintage"), (1800, "historic"),
])
def test_age_category_parametrized(year, expected):
    assert property_age_category(year, 2024) == expected
