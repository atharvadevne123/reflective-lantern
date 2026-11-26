"""Extended property features tests with parametrize and boundary coverage."""

from __future__ import annotations

import pytest

from app.features import (
    NEIGHBORHOODS,
    PROPERTY_TYPE_MULTIPLIERS,
    PROPERTY_TYPES,
    compute_gross_yield,
    compute_price_per_sqft,
    encode_property_features,
    property_type_to_multiplier,
    sqft_percentile,
)


@pytest.mark.parametrize(
    "price,sqft,expected",
    [
        (500_000.0, 1000.0, 500.0),
        (300_000.0, 1500.0, 200.0),
        (100_000.0, 500.0, 200.0),
        (0.0, 1000.0, 0.0),
    ],
)
def test_compute_price_per_sqft_parametrized(price: float, sqft: float, expected: float) -> None:
    assert compute_price_per_sqft(price, sqft) == pytest.approx(expected)


def test_compute_price_per_sqft_zero_sqft_returns_zero() -> None:
    assert compute_price_per_sqft(500_000.0, 0.0) == 0.0


def test_compute_price_per_sqft_negative_sqft_returns_zero() -> None:
    assert compute_price_per_sqft(500_000.0, -100.0) == 0.0


@pytest.mark.parametrize("property_type", PROPERTY_TYPES)
def test_encode_property_features_known_types(property_type: str) -> None:
    result = encode_property_features(property_type, "suburb")
    assert result["property_type_code"] >= 0


@pytest.mark.parametrize("neighborhood", NEIGHBORHOODS)
def test_encode_property_features_known_neighborhoods(neighborhood: str) -> None:
    result = encode_property_features("house", neighborhood)
    assert result["neighborhood_code"] >= 0


def test_encode_property_features_unknown_type_minus_one() -> None:
    result = encode_property_features("castle", "suburb")
    assert result["property_type_code"] == -1


def test_encode_property_features_unknown_neighborhood_minus_one() -> None:
    result = encode_property_features("house", "mars")
    assert result["neighborhood_code"] == -1


def test_encode_property_features_case_insensitive() -> None:
    result_lower = encode_property_features("house", "downtown")
    result_upper = encode_property_features("HOUSE", "DOWNTOWN")
    assert result_lower == result_upper


def test_encode_property_features_returns_dict_with_two_keys() -> None:
    result = encode_property_features("house", "suburb")
    assert set(result.keys()) == {"property_type_code", "neighborhood_code"}


@pytest.mark.parametrize(
    "annual_rental,price,expected",
    [
        (20_000.0, 400_000.0, 5.0),
        (10_000.0, 200_000.0, 5.0),
        (0.0, 300_000.0, 0.0),
        (15_000.0, 0.0, 0.0),
    ],
)
def test_compute_gross_yield_parametrized(
    annual_rental: float, price: float, expected: float
) -> None:
    result = compute_gross_yield(annual_rental, price)
    assert result == pytest.approx(expected, rel=1e-4)


def test_compute_gross_yield_negative_price_returns_zero() -> None:
    assert compute_gross_yield(10_000.0, -1.0) == 0.0


def test_sqft_percentile_empty_reference_returns_zero() -> None:
    assert sqft_percentile(1000.0, []) == 0.0


def test_sqft_percentile_all_smaller_returns_100() -> None:
    result = sqft_percentile(2000.0, [500.0, 800.0, 1000.0, 1500.0])
    assert result == 100.0


def test_sqft_percentile_all_larger_returns_zero() -> None:
    result = sqft_percentile(100.0, [500.0, 800.0, 1000.0])
    assert result == 0.0


@pytest.mark.parametrize(
    "sqft,reference,expected_range",
    [
        (500.0, [100.0, 300.0, 500.0, 700.0, 900.0], (30.0, 50.0)),
        (600.0, [100.0, 300.0, 500.0, 700.0, 900.0], (60.0, 80.0)),
    ],
)
def test_sqft_percentile_in_range(sqft: float, reference: list, expected_range: tuple) -> None:
    result = sqft_percentile(sqft, reference)
    assert expected_range[0] <= result <= expected_range[1]


@pytest.mark.parametrize(
    "property_type,expected_multiplier", list(PROPERTY_TYPE_MULTIPLIERS.items())
)
def test_property_type_to_multiplier_known(property_type: str, expected_multiplier: float) -> None:
    result = property_type_to_multiplier(property_type)
    assert result == pytest.approx(expected_multiplier)


def test_property_type_to_multiplier_unknown_returns_one() -> None:
    assert property_type_to_multiplier("castle") == 1.0


def test_property_type_to_multiplier_case_insensitive() -> None:
    assert property_type_to_multiplier("HOUSE") == property_type_to_multiplier("house")
