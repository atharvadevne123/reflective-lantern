"""Tests for app/data/features.py."""

from __future__ import annotations

import pytest


def test_price_ratio_basic():
    from app.data.features import price_ratio

    assert abs(price_ratio(100.0, 50.0) - 2.0) < 1e-9


def test_price_ratio_zero_reference():
    from app.data.features import price_ratio

    assert price_ratio(50.0, 0.0) == 1.0


def test_competitive_gap_positive():
    from app.data.features import competitive_gap

    gap = competitive_gap(110.0, 100.0)
    assert abs(gap - 0.1) < 1e-9


def test_competitive_gap_zero_competitor():
    from app.data.features import competitive_gap

    assert competitive_gap(100.0, 0.0) == 0.0


def test_build_feature_matrix(sample_records):
    from app.data.features import build_feature_matrix

    matrix = build_feature_matrix(sample_records)
    assert len(matrix) == len(sample_records)
    assert len(matrix[0]) == 5


def test_build_feature_matrix_first_row(sample_records):
    from app.data.features import build_feature_matrix

    matrix = build_feature_matrix(sample_records)
    rec = sample_records[0]
    assert matrix[0][0] == float(rec["base_price"])
    assert matrix[0][1] == float(rec["demand"])


@pytest.mark.parametrize(
    "price,reference,expected",
    [
        (100.0, 100.0, 1.0),
        (200.0, 100.0, 2.0),
        (50.0, 100.0, 0.5),
    ],
)
def test_price_ratio_parametrized(price: float, reference: float, expected: float) -> None:
    from app.data.features import price_ratio

    assert price_ratio(price, reference) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    "own_price,competitor_price",
    [
        (100.0, 90.0),
        (80.0, 100.0),
        (100.0, 100.0),
    ],
)
def test_competitive_gap_sign(own_price: float, competitor_price: float) -> None:
    from app.data.features import competitive_gap

    gap = competitive_gap(own_price, competitor_price)
    if own_price > competitor_price:
        assert gap > 0.0
    elif own_price < competitor_price:
        assert gap < 0.0
    else:
        assert gap == pytest.approx(0.0, abs=1e-6)


def test_build_feature_matrix_empty_returns_empty() -> None:
    from app.data.features import build_feature_matrix

    result = build_feature_matrix([])
    assert result == []


def test_build_feature_matrix_has_correct_width() -> None:
    from app.data.features import build_feature_matrix

    records = [
        {
            "base_price": 100.0,
            "demand": 50.0,
            "competition_price": 90.0,
            "day_of_week": 1,
            "is_weekend": False,
        }
    ]
    result = build_feature_matrix(records)
    assert len(result) == 1
    assert len(result[0]) == 5


def test_price_ratio_reference_zero_returns_one() -> None:
    from app.data.features import price_ratio

    assert price_ratio(100.0, 0.0) == 1.0


def test_competitive_gap_zero_competitor_returns_zero() -> None:
    from app.data.features import competitive_gap

    assert competitive_gap(100.0, 0.0) == 0.0


def test_price_momentum_empty_returns_empty() -> None:
    from app.data.features import price_momentum

    assert price_momentum([]) == []


def test_price_momentum_length_matches_input() -> None:
    from app.data.features import price_momentum

    prices = [10.0, 11.0, 12.0, 13.0, 14.0]
    result = price_momentum(prices, window=3)
    assert len(result) == len(prices)


def test_price_momentum_first_entries_zero() -> None:
    from app.data.features import price_momentum

    prices = [10.0, 20.0, 30.0, 40.0]
    result = price_momentum(prices, window=3)
    assert result[0] == 0.0
    assert result[1] == 0.0
    assert result[2] == 0.0


def test_feature_names_constant_is_list() -> None:
    from app.data.features import FEATURE_NAMES

    assert isinstance(FEATURE_NAMES, list)
    assert len(FEATURE_NAMES) == 5


def test_demand_elasticity_feature_no_price_change_returns_zero() -> None:
    from app.data.features import demand_elasticity_feature

    result = demand_elasticity_feature(100.0, 50.0, 50.0)
    assert result == 0.0


def test_demand_elasticity_feature_zero_base_price_returns_zero() -> None:
    from app.data.features import demand_elasticity_feature

    result = demand_elasticity_feature(100.0, 50.0, 0.0)
    assert result == 0.0
