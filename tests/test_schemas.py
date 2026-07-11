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


@pytest.mark.parametrize(
    "score_field,valid_val",
    [
        ("school_score", 1.0),
        ("school_score", 10.0),
        ("transit_score", 5.0),
        ("walkability_score", 7.5),
    ],
)
def test_score_fields_valid_range(score_field, valid_val) -> None:
    props = base_props()
    props[score_field] = valid_val
    prop = PropertyInput(**props)
    assert getattr(prop, score_field) == valid_val


@pytest.mark.parametrize("bedrooms", [1, 3, 5, 10])
def test_bedrooms_valid_range(bedrooms) -> None:
    props = base_props()
    props["bedrooms"] = bedrooms
    prop = PropertyInput(**props)
    assert prop.bedrooms == bedrooms


def test_batch_input_single_property() -> None:
    body = BatchPropertyInput(properties=[PropertyInput(**base_props())])
    assert len(body.properties) == 1


@pytest.mark.parametrize("bathrooms", [1.0, 1.5, 2.0, 3.5])
def test_bathrooms_valid_range(bathrooms) -> None:
    props = base_props()
    props["bathrooms"] = bathrooms
    prop = PropertyInput(**props)
    assert prop.bathrooms == bathrooms


def test_state_and_city_optional() -> None:
    props = base_props()
    prop = PropertyInput(**props)
    assert prop.state == ""
    assert prop.city == ""


def test_max_sqft_constant_used_as_upper_bound() -> None:
    from app.schemas import _MAX_SQFT

    props = base_props()
    props["sqft"] = float(_MAX_SQFT)
    prop = PropertyInput(**props)
    assert prop.sqft == float(_MAX_SQFT)


def test_sqft_exceeds_max_sqft_constant_raises() -> None:
    from app.schemas import _MAX_SQFT

    props = base_props()
    props["sqft"] = float(_MAX_SQFT) + 1.0
    with pytest.raises(ValidationError):
        PropertyInput(**props)


def test_batch_max_constant_allows_exactly_100() -> None:
    from app.schemas import _BATCH_MAX

    props = [PropertyInput(**base_props())] * _BATCH_MAX
    body = BatchPropertyInput(properties=props)
    assert len(body.properties) == _BATCH_MAX


def test_year_built_min_constant_boundary() -> None:
    from app.schemas import _MIN_YEAR

    props = base_props()
    props["year_built"] = _MIN_YEAR
    prop = PropertyInput(**props)
    assert prop.year_built == _MIN_YEAR


@pytest.mark.parametrize("bedrooms", [1, 10, 20])
def test_bedrooms_max_constant_boundary(bedrooms: int) -> None:
    props = base_props()
    props["bedrooms"] = bedrooms
    prop = PropertyInput(**props)
    assert prop.bedrooms == bedrooms


def test_bedrooms_above_max_raises() -> None:
    from app.schemas import _MAX_BEDROOMS

    props = base_props()
    props["bedrooms"] = _MAX_BEDROOMS + 1
    with pytest.raises(ValidationError):
        PropertyInput(**props)


def test_min_year_constant() -> None:
    from app.schemas import _MIN_YEAR

    assert _MIN_YEAR < 1900


def test_max_year_constant() -> None:
    from app.schemas import _MAX_YEAR

    assert _MAX_YEAR >= 2024


def test_max_top_k_constant() -> None:
    from app.schemas import _MAX_TOP_K

    assert _MAX_TOP_K > 0


@pytest.mark.parametrize("year", [1850, 1920, 1970, 2000, 2024])
def test_valid_year_built_accepted(year: int) -> None:
    props = base_props()
    props["year_built"] = year
    prop = PropertyInput(**props)
    assert prop.year_built == year


@pytest.mark.parametrize("sqft", [500.0, 1000.0, 2500.0, 5000.0])
def test_valid_sqft_values(sqft: float) -> None:
    props = base_props()
    props["sqft"] = sqft
    prop = PropertyInput(**props)
    assert prop.sqft == sqft


@pytest.mark.parametrize("bathrooms", [1.0, 1.5, 2.0, 3.5])
def test_valid_bathroom_values(bathrooms: float) -> None:
    props = base_props()
    props["bathrooms"] = bathrooms
    prop = PropertyInput(**props)
    assert prop.bathrooms == bathrooms
