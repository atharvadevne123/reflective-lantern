"""Tests for app/stats_utils.py."""

from __future__ import annotations

import pytest


def test_mae_perfect() -> None:
    from app.stats_utils import mean_absolute_error
    assert mean_absolute_error([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_mae_basic() -> None:
    from app.stats_utils import mean_absolute_error
    result = mean_absolute_error([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    assert abs(result - 1.0) < 1e-6


def test_mae_empty_raises() -> None:
    from app.stats_utils import mean_absolute_error
    with pytest.raises(ValueError):
        mean_absolute_error([], [])


def test_mae_mismatched_raises() -> None:
    from app.stats_utils import mean_absolute_error
    with pytest.raises(ValueError):
        mean_absolute_error([1.0, 2.0], [1.0])


def test_rmse_perfect() -> None:
    from app.stats_utils import root_mean_squared_error
    assert root_mean_squared_error([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_rmse_basic() -> None:
    from app.stats_utils import root_mean_squared_error
    result = root_mean_squared_error([0.0, 0.0], [1.0, 1.0])
    assert abs(result - 1.0) < 1e-6


def test_rmse_empty_raises() -> None:
    from app.stats_utils import root_mean_squared_error
    with pytest.raises(ValueError):
        root_mean_squared_error([], [])


def test_r_squared_perfect() -> None:
    from app.stats_utils import r_squared
    actual = [1.0, 2.0, 3.0, 4.0]
    assert abs(r_squared(actual, actual) - 1.0) < 1e-6


def test_r_squared_no_better_than_mean() -> None:
    from app.stats_utils import r_squared
    actual = [1.0, 2.0, 3.0]
    predicted = [2.0, 2.0, 2.0]  # constant = mean
    result = r_squared(actual, predicted)
    assert abs(result) < 0.01


def test_r_squared_empty_raises() -> None:
    from app.stats_utils import r_squared
    with pytest.raises(ValueError):
        r_squared([], [])


def test_mape_basic() -> None:
    from app.stats_utils import mape
    result = mape([100.0, 200.0], [110.0, 190.0])
    assert result > 0


def test_mape_zero_actual_raises() -> None:
    from app.stats_utils import mape
    with pytest.raises(ValueError, match="undefined"):
        mape([0.0, 1.0], [1.0, 1.0])


def test_mape_empty_raises() -> None:
    from app.stats_utils import mape
    with pytest.raises(ValueError):
        mape([], [])


def test_coefficient_of_variation_flat() -> None:
    from app.stats_utils import coefficient_of_variation
    assert coefficient_of_variation([5.0] * 10) == 0.0


def test_coefficient_of_variation_nonzero() -> None:
    from app.stats_utils import coefficient_of_variation
    result = coefficient_of_variation([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result > 0


def test_coefficient_of_variation_too_short() -> None:
    from app.stats_utils import coefficient_of_variation
    assert coefficient_of_variation([5.0]) == 0.0


def test_percentile_median() -> None:
    from app.stats_utils import percentile
    result = percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50)
    assert abs(result - 3.0) < 0.01


def test_percentile_min() -> None:
    from app.stats_utils import percentile
    result = percentile([1.0, 2.0, 3.0], 0)
    assert result == 1.0


def test_percentile_max() -> None:
    from app.stats_utils import percentile
    result = percentile([1.0, 2.0, 3.0], 100)
    assert result == 3.0


def test_percentile_empty_raises() -> None:
    from app.stats_utils import percentile
    with pytest.raises(ValueError, match="must not be empty"):
        percentile([], 50)


def test_percentile_invalid_p_raises() -> None:
    from app.stats_utils import percentile
    with pytest.raises(ValueError, match="p must be"):
        percentile([1.0, 2.0], 101)


@pytest.mark.parametrize("p", [0, 25, 50, 75, 100])
def test_percentile_various_p(p) -> None:
    from app.stats_utils import percentile
    values = list(range(1, 101))
    result = percentile(values, p)
    assert 1 <= result <= 100


@pytest.mark.parametrize("actual,predicted,expected_mae", [
    ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0),
    ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 0.0),
    ([10.0], [8.0], 2.0),
])
def test_mae_parametrized(actual: list, predicted: list, expected_mae: float) -> None:
    from app.stats_utils import mean_absolute_error
    assert mean_absolute_error(actual, predicted) == pytest.approx(expected_mae, rel=1e-4)


@pytest.mark.parametrize("actual,predicted", [
    ([1.0, 2.0], [1.0]),
    ([1.0], [1.0, 2.0]),
])
def test_mae_length_mismatch_raises(actual: list, predicted: list) -> None:
    from app.stats_utils import mean_absolute_error
    with pytest.raises(ValueError):
        mean_absolute_error(actual, predicted)


@pytest.mark.parametrize("p,expected_percentile", [
    (0.0, 1.0),
    (50.0, 3.0),
    (100.0, 5.0),
])
def test_percentile_parametrized(p: float, expected_percentile: float) -> None:
    from app.stats_utils import percentile
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = percentile(values, p)
    assert result == pytest.approx(expected_percentile, abs=1.0)


def test_coefficient_of_variation_zero_mean() -> None:
    from app.stats_utils import coefficient_of_variation
    assert coefficient_of_variation([0.0, 0.0, 0.0]) == 0.0


def test_r_squared_perfect_fit() -> None:
    from app.stats_utils import r_squared
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert r_squared(values, values) == pytest.approx(1.0)


def test_mape_raises_on_zero_actual() -> None:
    from app.stats_utils import mape
    with pytest.raises(ValueError, match="zero"):
        mape([0.0, 1.0], [0.5, 1.0])
