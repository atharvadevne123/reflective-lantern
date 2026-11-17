"""Tests for Property-Sage input validators."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.features import NEIGHBORHOODS, PROPERTY_TYPES
from app.validators import (
    validate_neighborhood_value,
    validate_property_type_value,
    validate_sqft_vs_bedrooms,
)


def test_validate_neighborhood_valid():
    sample = next(iter(NEIGHBORHOODS))
    result = validate_neighborhood_value(sample)
    assert result == sample.lower().strip()


def test_validate_neighborhood_case_insensitive():
    sample = next(iter(NEIGHBORHOODS))
    result = validate_neighborhood_value(sample.upper())
    assert result == sample.lower()


def test_validate_neighborhood_invalid_raises():
    with pytest.raises(ValueError):
        validate_neighborhood_value("atlantis_xyz")


def test_validate_neighborhood_strips_whitespace():
    sample = next(iter(NEIGHBORHOODS))
    result = validate_neighborhood_value(f"  {sample}  ")
    assert result == sample.lower()


def test_validate_property_type_valid():
    sample = next(iter(PROPERTY_TYPES))
    result = validate_property_type_value(sample)
    assert result == sample.lower().strip()


def test_validate_property_type_invalid_raises():
    with pytest.raises(ValueError):
        validate_property_type_value("space_station")


def test_validate_property_type_case_insensitive():
    sample = next(iter(PROPERTY_TYPES))
    result = validate_property_type_value(sample.upper())
    assert result == sample.lower()


def test_validate_sqft_vs_bedrooms_valid():
    validate_sqft_vs_bedrooms(1000.0, 3)  # ~333 sqft/bedroom, should be fine


def test_validate_sqft_vs_bedrooms_invalid_ratio():
    with pytest.raises(ValueError):
        validate_sqft_vs_bedrooms(100.0, 5)  # 20 sqft/bedroom, too low


def test_validate_sqft_vs_bedrooms_zero_bedrooms_treated_as_one():
    validate_sqft_vs_bedrooms(200.0, 0)  # 200 sqft / 1 = 200, should pass


@pytest.mark.parametrize("hood", list(NEIGHBORHOODS)[:5])
def test_all_valid_neighborhoods(hood):
    result = validate_neighborhood_value(hood)
    assert result in NEIGHBORHOODS


@pytest.mark.parametrize("ptype", list(PROPERTY_TYPES)[:5])
def test_all_valid_property_types(ptype):
    result = validate_property_type_value(ptype)
    assert result in PROPERTY_TYPES
