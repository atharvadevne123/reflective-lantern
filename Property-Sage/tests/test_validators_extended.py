"""Extended tests for Property-Sage app/validators.py."""

from __future__ import annotations

import pytest

from app.validators import (
    validate_neighborhood_value,
    validate_property_type_value,
    validate_sqft_vs_bedrooms,
)


class TestValidateNeighborhoodValue:
    def test_valid_neighborhood_lowercase(self) -> None:
        assert validate_neighborhood_value("downtown") == "downtown"

    def test_valid_neighborhood_uppercase_normalised(self) -> None:
        assert validate_neighborhood_value("DOWNTOWN") == "downtown"

    def test_valid_neighborhood_with_spaces(self) -> None:
        assert validate_neighborhood_value("  suburb  ") == "suburb"

    def test_invalid_neighborhood_raises(self) -> None:
        with pytest.raises(ValueError, match="neighbourhood must be one of"):
            validate_neighborhood_value("mars")

    def test_all_known_neighborhoods_pass(self) -> None:
        from app.features import NEIGHBORHOODS
        for n in NEIGHBORHOODS:
            assert validate_neighborhood_value(n) == n

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_neighborhood_value("")


class TestValidatePropertyTypeValue:
    def test_valid_apartment(self) -> None:
        assert validate_property_type_value("apartment") == "apartment"

    def test_case_insensitive(self) -> None:
        assert validate_property_type_value("CONDO") == "condo"

    def test_strips_whitespace(self) -> None:
        assert validate_property_type_value("  house  ") == "house"

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="property_type must be one of"):
            validate_property_type_value("castle")

    def test_all_known_types_pass(self) -> None:
        from app.features import PROPERTY_TYPES
        for pt in PROPERTY_TYPES:
            assert validate_property_type_value(pt) == pt


class TestValidateSqftVsBedrooms:
    def test_valid_ratio_no_error(self) -> None:
        validate_sqft_vs_bedrooms(1000.0, 3)  # 333 sqft/bed

    def test_borderline_valid_exactly_80(self) -> None:
        validate_sqft_vs_bedrooms(80.0, 1)  # exactly 80

    def test_below_minimum_raises(self) -> None:
        with pytest.raises(ValueError, match="sqft_per_bedroom"):
            validate_sqft_vs_bedrooms(79.0, 1)

    def test_zero_bedrooms_uses_max_1(self) -> None:
        validate_sqft_vs_bedrooms(500.0, 0)  # ratio = 500/1 = 500, OK

    def test_very_large_sqft_no_error(self) -> None:
        validate_sqft_vs_bedrooms(10000.0, 5)

    def test_low_sqft_many_bedrooms_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_sqft_vs_bedrooms(200.0, 5)  # 40 sqft/bed
