"""Extended suite-cast feature tests with parametrize and boundary coverage."""
from __future__ import annotations

import pytest

from app.features import (
    adr_from_revenue,
    competitor_rate_ratio,
    lead_time_bucket,
    length_of_stay_bucket,
    length_of_stay_mix,
    occupancy_yoy_delta,
    revpar,
)


@pytest.mark.parametrize("lead_days,expected_bucket", [
    (0, 0),
    (3, 0),
    (4, 1),
    (14, 1),
    (15, 2),
    (60, 2),
    (61, 3),
    (365, 3),
])
def test_lead_time_bucket_parametrized(lead_days: int, expected_bucket: int) -> None:
    assert lead_time_bucket(lead_days) == expected_bucket


@pytest.mark.parametrize("room_rate,competitor_avg,expected_range", [
    (100.0, 100.0, (0.9, 1.1)),
    (50.0, 100.0, (0.49, 0.51)),
    (500.0, 100.0, (2.9, 3.1)),
    (100.0, 0.0, (0.5, 0.75)),
])
def test_competitor_rate_ratio_parametrized(
    room_rate: float, competitor_avg: float, expected_range: tuple
) -> None:
    result = competitor_rate_ratio(room_rate, competitor_avg)
    assert expected_range[0] <= result <= expected_range[1]


def test_competitor_rate_ratio_clamped_min() -> None:
    result = competitor_rate_ratio(1.0, 1000.0)
    assert result == pytest.approx(0.5)


def test_competitor_rate_ratio_clamped_max() -> None:
    result = competitor_rate_ratio(1000.0, 10.0)
    assert result == pytest.approx(3.0)


@pytest.mark.parametrize("current,prev,expected_range", [
    (0.9, 0.5, (0.35, 0.45)),
    (0.5, 0.9, (-0.45, -0.35)),
    (1.0, 0.0, (0.9, 1.0)),
    (0.0, 1.0, (-1.0, -0.9)),
])
def test_occupancy_yoy_delta_parametrized(
    current: float, prev: float, expected_range: tuple
) -> None:
    result = occupancy_yoy_delta(current, prev)
    assert expected_range[0] <= result <= expected_range[1]


def test_occupancy_yoy_delta_clamped_at_plus_one() -> None:
    result = occupancy_yoy_delta(1.0, -0.5)
    assert result == pytest.approx(1.0)


def test_occupancy_yoy_delta_clamped_at_minus_one() -> None:
    result = occupancy_yoy_delta(-0.5, 1.0)
    assert result == pytest.approx(-1.0)


@pytest.mark.parametrize("revenue,rooms,expected", [
    (10000.0, 100, 100.0),
    (0.0, 100, 0.0),
    (5000.0, 0, 0.0),
])
def test_adr_from_revenue_parametrized(revenue: float, rooms: int, expected: float) -> None:
    assert adr_from_revenue(revenue, rooms) == pytest.approx(expected)


@pytest.mark.parametrize("revenue,available_rooms,expected", [
    (10000.0, 100, 100.0),
    (0.0, 100, 0.0),
    (5000.0, 0, 0.0),
])
def test_revpar_parametrized(revenue: float, available_rooms: int, expected: float) -> None:
    assert revpar(revenue, available_rooms) == pytest.approx(expected)


@pytest.mark.parametrize("nights,expected_bucket", [
    (1, "short"),
    (2, "short"),
    (3, "medium"),
    (6, "medium"),
    (7, "long"),
    (30, "long"),
])
def test_length_of_stay_bucket_parametrized(nights: int, expected_bucket: str) -> None:
    assert length_of_stay_bucket(nights) == expected_bucket


def test_length_of_stay_mix_empty_returns_zeros() -> None:
    result = length_of_stay_mix([])
    assert result == {"one_night": 0.0, "two_three_nights": 0.0, "four_plus_nights": 0.0}


def test_length_of_stay_mix_all_one_night() -> None:
    result = length_of_stay_mix([1, 1, 1])
    assert result["one_night"] == pytest.approx(1.0)
    assert result["two_three_nights"] == pytest.approx(0.0)


def test_length_of_stay_mix_sums_to_one() -> None:
    result = length_of_stay_mix([1, 2, 3, 7, 1, 5, 2])
    total = result["one_night"] + result["two_three_nights"] + result["four_plus_nights"]
    assert total == pytest.approx(1.0, abs=1e-6)
