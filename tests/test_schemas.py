"""Tests for Pydantic request/response schemas."""

import pytest
from pydantic import ValidationError

from app.schemas import BatchPropertyInput, PropertyInput


def base_props() -> dict:
    return {
        "sqft": 1800.0,
        "bedrooms": 3,
        "bathrooms": 2.0,
        "lot_size": 5000.0,
        "year_built": 1990,
        "condition_score": 7.5,
        "zipcode": "94102",
        "school_score": 8.0,
        "transit_score": 9.0,
        "walkability_score": 8.5,
        "crime_rate": 0.3,
        "median_neighborhood_price": 1_200_000.0,
        "median_price_per_sqft": 800.0,
        "avg_rental_yield": 0.05,
        "listing_days": 14,
    }


def test_valid_property_input() -> None:
    prop = PropertyInput(**base_props())
    assert prop.sqft == 1800.0
    assert prop.bedrooms == 3


@pytest.mark.parametrize(
    "field,bad_val",
    [
        ("sqft", -1),
        ("sqft", 0),
        ("bedrooms", 0),
        ("bedrooms", 25),
        ("bathrooms", 0),
        ("year_built", 1700),
        ("year_built", 2030),
        ("crime_rate", -0.1),
        ("crime_rate", 1.5),
        ("condition_score", 0.5),
        ("condition_score", 11.0),
        ("avg_rental_yield", -0.1),
    ],
)
def test_invalid_field_raises_validation_error(field, bad_val) -> None:
    props = base_props()
    props[field] = bad_val
    with pytest.raises(ValidationError):
        PropertyInput(**props)


def test_renovation_year_before_built_raises() -> None:
    props = base_props()
    props["renovation_year"] = 1980
    props["year_built"] = 1990
    with pytest.raises(ValidationError):
        PropertyInput(**props)


def test_renovation_year_equal_to_built_is_valid() -> None:
    props = base_props()
    props["renovation_year"] = 1990
    prop = PropertyInput(**props)
    assert prop.renovation_year == 1990


def test_zipcode_too_short_raises() -> None:
    props = base_props()
    props["zipcode"] = "123"
    with pytest.raises(ValidationError):
        PropertyInput(**props)


def test_batch_input_empty_raises() -> None:
    with pytest.raises(ValidationError):
        BatchPropertyInput(properties=[])


def test_batch_input_too_many_raises() -> None:
    props = [PropertyInput(**base_props())] * 101
    with pytest.raises(ValidationError):
        BatchPropertyInput(properties=props)


def test_batch_input_valid() -> None:
    body = BatchPropertyInput(properties=[PropertyInput(**base_props())] * 5)
    assert len(body.properties) == 5


def test_optional_fields_default() -> None:
    props = base_props()
    prop = PropertyInput(**props)
    assert prop.renovation_year is None
    assert prop.list_price is None
    assert prop.city == ""


def test_renovation_year_before_built_by_ten_years_raises() -> None:
    props = base_props()
    props["year_built"] = 2000
    props["renovation_year"] = 1990
    with pytest.raises(ValidationError):
        PropertyInput(**props)


def test_renovation_year_same_as_built_is_valid() -> None:
    props = base_props()
    props["year_built"] = 2000
    props["renovation_year"] = 2000
    prop = PropertyInput(**props)
    assert prop.renovation_year == 2000


def test_sqft_upper_bound_valid() -> None:
    props = base_props()
    props["sqft"] = 50_000.0
    prop = PropertyInput(**props)
    assert prop.sqft == 50_000.0


def test_sqft_exceeds_upper_bound_raises() -> None:
    props = base_props()
    props["sqft"] = 50_001.0
    with pytest.raises(ValidationError):
        PropertyInput(**props)


def test_list_price_zero_raises() -> None:
    props = base_props()
    props["list_price"] = 0.0
    with pytest.raises(ValidationError):
        PropertyInput(**props)


def test_listing_days_upper_bound_valid() -> None:
    props = base_props()
    props["listing_days"] = 3650
    prop = PropertyInput(**props)
    assert prop.listing_days == 3650
