"""Tests for encode_property_features."""

import pytest

from app.features import encode_property_features


def test_known_values():
    result = encode_property_features("house", "suburb")
    assert result["property_type_code"] >= 0
    assert result["neighborhood_code"] >= 0


def test_unknown_type():
    result = encode_property_features("castle", "suburb")
    assert result["property_type_code"] == -1


def test_unknown_neighborhood():
    result = encode_property_features("house", "moon_base")
    assert result["neighborhood_code"] == -1


@pytest.mark.parametrize("pt", ["apartment", "house", "condo", "townhouse", "studio"])
def test_all_property_types(pt):
    result = encode_property_features(pt, "suburb")
    assert result["property_type_code"] >= 0
