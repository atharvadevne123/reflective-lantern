"""Tests for property_type_to_multiplier."""

import pytest

from app.features import property_type_to_multiplier


def test_house():
    assert property_type_to_multiplier("house") == 1.0


def test_studio_discounted():
    assert property_type_to_multiplier("studio") < 1.0


def test_unknown_defaults_to_one():
    assert property_type_to_multiplier("castle") == 1.0


def test_case_insensitive():
    assert property_type_to_multiplier("CONDO") == property_type_to_multiplier("condo")


@pytest.mark.parametrize("pt", ["house", "condo", "apartment", "townhouse", "studio"])
def test_all_known_types_lt_or_eq_one(pt):
    assert property_type_to_multiplier(pt) <= 1.0
