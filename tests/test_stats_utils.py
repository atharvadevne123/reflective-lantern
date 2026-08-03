"""Tests for app/stats_utils.py."""

from __future__ import annotations

import pytest

from app.stats_utils import (
    geometric_mean,
    harmonic_mean,
    interquartile_range,
    normalize_series,
    percentile_rank,
    weighted_average,
    zscore,
)


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


@pytest.mark.parametrize(
    "actual,predicted,expected_mae",
    [
        ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 0.0),
        ([10.0], [8.0], 2.0),
    ],
)
def test_mae_parametrized(actual: list, predicted: list, expected_mae: float) -> None:
    from app.stats_utils import mean_absolute_error

    assert mean_absolute_error(actual, predicted) == pytest.approx(expected_mae, rel=1e-4)


@pytest.mark.parametrize(
    "actual,predicted",
    [
        ([1.0, 2.0], [1.0]),
        ([1.0], [1.0, 2.0]),
    ],
)
def test_mae_length_mismatch_raises(actual: list, predicted: list) -> None:
    from app.stats_utils import mean_absolute_error

    with pytest.raises(ValueError):
        mean_absolute_error(actual, predicted)


@pytest.mark.parametrize(
    "p,expected_percentile",
    [
        (0.0, 1.0),
        (50.0, 3.0),
        (100.0, 5.0),
    ],
)
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


def test_geometric_mean_basic() -> None:
    assert geometric_mean([1.0, 4.0]) == pytest.approx(2.0, rel=1e-4)


def test_geometric_mean_single() -> None:
    assert geometric_mean([5.0]) == pytest.approx(5.0)


def test_geometric_mean_empty_raises() -> None:
    with pytest.raises(ValueError):
        geometric_mean([])


def test_geometric_mean_negative_raises() -> None:
    with pytest.raises(ValueError):
        geometric_mean([1.0, -2.0])


def test_harmonic_mean_basic() -> None:
    result = harmonic_mean([1.0, 2.0, 4.0])
    assert result == pytest.approx(12.0 / 7.0, rel=1e-4)


def test_harmonic_mean_single() -> None:
    assert harmonic_mean([8.0]) == pytest.approx(8.0)


def test_harmonic_mean_empty_raises() -> None:
    with pytest.raises(ValueError):
        harmonic_mean([])


def test_harmonic_mean_zero_raises() -> None:
    with pytest.raises(ValueError):
        harmonic_mean([0.0, 1.0])


def test_weighted_average_basic() -> None:
    result = weighted_average([10.0, 20.0], [1.0, 1.0])
    assert result == pytest.approx(15.0)


def test_weighted_average_unequal_weights() -> None:
    result = weighted_average([10.0, 30.0], [3.0, 1.0])
    assert result == pytest.approx(15.0, rel=1e-4)


def test_weighted_average_empty_raises() -> None:
    with pytest.raises(ValueError):
        weighted_average([], [])


def test_weighted_average_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError):
        weighted_average([1.0, 2.0], [1.0])


@pytest.mark.parametrize(
    "vals,expected",
    [
        ([2.0, 8.0], 4.0),
        ([1.0, 1.0, 1.0], 1.0),
    ],
)
def test_geometric_mean_parametrized(vals: list, expected: float) -> None:
    assert geometric_mean(vals) == pytest.approx(expected, rel=1e-4)


def test_normalize_series_basic() -> None:
    result = normalize_series([0.0, 5.0, 10.0])
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.5)


def test_normalize_series_constant() -> None:
    assert normalize_series([7.0, 7.0, 7.0]) == [0.0, 0.0, 0.0]


def test_normalize_series_empty_raises() -> None:
    with pytest.raises(ValueError):
        normalize_series([])


def test_normalize_series_length_preserved() -> None:
    vals = [1.0, 3.0, 7.0, 2.0, 9.0]
    result = normalize_series(vals)
    assert len(result) == len(vals)


def test_interquartile_range_basic() -> None:
    result = interquartile_range([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result == pytest.approx(2.0, rel=1e-4)


def test_interquartile_range_single() -> None:
    assert interquartile_range([42.0]) == 0.0


def test_interquartile_range_empty_raises() -> None:
    with pytest.raises(ValueError):
        interquartile_range([])


@pytest.mark.parametrize(
    "values,expected_iqr",
    [
        ([1.0, 2.0, 3.0, 4.0], pytest.approx(1.5, rel=0.1)),
        ([10.0, 10.0, 10.0, 10.0], 0.0),
    ],
)
def test_interquartile_range_parametrized(values: list, expected_iqr: object) -> None:
    assert interquartile_range(values) == expected_iqr


def test_percentile_rank_minimum() -> None:
    result = percentile_rank([1.0, 2.0, 3.0, 4.0, 5.0], 1.0)
    assert result == pytest.approx(20.0)


def test_percentile_rank_maximum() -> None:
    result = percentile_rank([1.0, 2.0, 3.0, 4.0, 5.0], 5.0)
    assert result == pytest.approx(100.0)


def test_percentile_rank_empty_raises() -> None:
    with pytest.raises(ValueError):
        percentile_rank([], 5.0)


def test_percentile_rank_above_all() -> None:
    result = percentile_rank([1.0, 2.0, 3.0], 10.0)
    assert result == pytest.approx(100.0)


def test_percentile_rank_below_all() -> None:
    result = percentile_rank([5.0, 6.0, 7.0], 0.0)
    assert result == pytest.approx(0.0)


@pytest.mark.parametrize(
    "target,expected_pct",
    [
        (1.0, 20.0),
        (3.0, 60.0),
        (5.0, 100.0),
    ],
)
def test_percentile_rank_parametrized(target: float, expected_pct: float) -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile_rank(values, target) == pytest.approx(expected_pct)


class TestZscore:
    def test_mean_element_is_zero(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert zscore(vals, 3.0) == pytest.approx(0.0, abs=1e-6)

    def test_above_mean_is_positive(self) -> None:
        vals = [0.0, 0.0, 0.0, 0.0, 10.0]
        assert zscore(vals, 10.0) > 0

    def test_below_mean_is_negative(self) -> None:
        vals = [5.0, 10.0, 15.0]
        assert zscore(vals, 5.0) < 0

    def test_constant_series_returns_zero(self) -> None:
        vals = [4.0, 4.0, 4.0]
        assert zscore(vals, 4.0) == 0.0

    def test_empty_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            zscore([], 1.0)

    def test_known_zscore(self) -> None:
        # mean=0, std=1 → zscore of 2 is 2.0
        vals = [-1.0, 0.0, 1.0]
        result = zscore(vals, 0.0)
        assert result == pytest.approx(0.0, abs=1e-4)


def test_weighted_average_equal_weights() -> None:
    from app.stats_utils import weighted_average

    result = weighted_average([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
    assert result == pytest.approx(2.0)


def test_weighted_average_skewed_weights() -> None:
    from app.stats_utils import weighted_average

    result = weighted_average([0.0, 10.0], [1.0, 9.0])
    assert result == pytest.approx(9.0)


def test_weighted_average_empty_raises_inline() -> None:
    from app.stats_utils import weighted_average

    with pytest.raises(ValueError):
        weighted_average([], [])


def test_harmonic_mean_is_float() -> None:
    from app.stats_utils import harmonic_mean

    result = harmonic_mean([1.0, 2.0, 4.0])
    assert isinstance(result, float)
    assert result > 0


def test_geometric_mean_two_values() -> None:
    from app.stats_utils import geometric_mean

    result = geometric_mean([1.0, 4.0])
    assert result == pytest.approx(2.0, rel=1e-4)


@pytest.mark.parametrize(
    "values,expected",
    [
        ([1.0, 1.0, 1.0], 1.0),
        ([2.0, 8.0], 4.0),
    ],
)
def test_geometric_mean_known_values(values, expected) -> None:
    from app.stats_utils import geometric_mean

    assert geometric_mean(values) == pytest.approx(expected, rel=1e-4)


class TestCoefficientOfVariation:
    def test_identical_values_returns_zero(self) -> None:
        from app.stats_utils import coefficient_of_variation

        assert coefficient_of_variation([5.0] * 10) == 0.0

    def test_known_cv(self) -> None:
        from app.stats_utils import coefficient_of_variation

        # Values 0,2,4: mean=2, std=sqrt(8/3)≈1.633
        result = coefficient_of_variation([0.0, 2.0, 4.0])
        assert result == pytest.approx(81.6497, rel=1e-3)

    def test_empty_raises(self) -> None:
        import pytest as _pytest

        from app.stats_utils import coefficient_of_variation

        with _pytest.raises(ValueError):
            coefficient_of_variation([])

    def test_zero_mean_raises(self) -> None:
        import pytest as _pytest

        from app.stats_utils import coefficient_of_variation

        with _pytest.raises(ValueError, match="near zero"):
            coefficient_of_variation([0.0, 0.0, 0.0])

    @pytest.mark.parametrize("vals", [[1.0, 2.0, 3.0], [10.0, 20.0]])
    def test_returns_positive_float(self, vals: list) -> None:
        from app.stats_utils import coefficient_of_variation

        result = coefficient_of_variation(vals)
        assert result > 0.0


class TestExponentialMovingAverage:
    def test_single_element(self) -> None:
        from app.stats_utils import exponential_moving_average

        assert exponential_moving_average([5.0]) == [5.0]

    def test_alpha_1_equals_input(self) -> None:
        from app.stats_utils import exponential_moving_average

        vals = [1.0, 2.0, 3.0, 4.0]
        result = exponential_moving_average(vals, alpha=1.0)
        assert result == pytest.approx(vals, abs=1e-5)

    def test_same_length_as_input(self) -> None:
        from app.stats_utils import exponential_moving_average

        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert len(exponential_moving_average(vals)) == len(vals)

    def test_invalid_alpha_raises(self) -> None:
        import pytest as _pytest

        from app.stats_utils import exponential_moving_average

        with _pytest.raises(ValueError):
            exponential_moving_average([1.0, 2.0], alpha=0.0)


class TestWinsorize:
    def test_no_clipping_needed(self) -> None:
        from app.stats_utils import winsorize

        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = winsorize(vals, lower_pct=0.0, upper_pct=100.0)
        assert result == vals

    def test_clips_extremes(self) -> None:
        from app.stats_utils import winsorize

        vals = [0.0, 1.0, 2.0, 3.0, 100.0]
        result = winsorize(vals, lower_pct=10.0, upper_pct=90.0)
        assert max(result) < 100.0

    def test_same_length(self) -> None:
        from app.stats_utils import winsorize

        vals = list(range(20))
        assert len(winsorize(vals)) == len(vals)

    @pytest.mark.parametrize("lower,upper", [(5.0, 95.0), (25.0, 75.0)])
    def test_result_within_bounds(self, lower: float, upper: float) -> None:
        from app.stats_utils import winsorize

        vals = list(range(100))
        result = winsorize(vals, lower_pct=lower, upper_pct=upper)
        assert min(result) >= vals[int(lower)]
