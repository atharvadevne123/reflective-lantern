"""Schema validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import AnomalyRequest, DriftRequest, EnergyReadingIn


def test_energy_reading_valid():
    r = EnergyReadingIn(
        building_id="bldg-001",
        timestamp="2025-06-01T14:00:00",
        hour=14,
        day_of_week=1,
        month=6,
        temperature_c=28.5,
        humidity_pct=60.0,
        occupancy=50,
        hvac_state=1,
        consumption_kwh=12.0,
    )
    assert r.building_id == "bldg-001"
    assert r.hour == 14


def test_energy_reading_invalid_hour():
    with pytest.raises(ValidationError):
        EnergyReadingIn(
            building_id="b",
            timestamp="2025-01-01T00:00:00",
            hour=25,
            day_of_week=0,
            month=1,
            temperature_c=20.0,
            humidity_pct=50.0,
            occupancy=0,
            hvac_state=0,
        )


def test_energy_reading_invalid_building_id():
    with pytest.raises(ValidationError):
        EnergyReadingIn(
            building_id="bad id!@#",
            timestamp="2025-01-01T00:00:00",
            hour=12,
            day_of_week=0,
            month=1,
            temperature_c=20.0,
            humidity_pct=50.0,
            occupancy=0,
            hvac_state=0,
        )


@pytest.mark.parametrize("temp", [-41.0, 61.0])
def test_energy_reading_invalid_temperature(temp):
    with pytest.raises(ValidationError):
        EnergyReadingIn(
            building_id="b001",
            timestamp="2025-01-01T00:00:00",
            hour=12,
            day_of_week=0,
            month=1,
            temperature_c=temp,
            humidity_pct=50.0,
            occupancy=0,
            hvac_state=0,
        )


def test_drift_request_min_length():
    with pytest.raises(ValidationError):
        DriftRequest(current_values=[1.0, 2.0])


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
