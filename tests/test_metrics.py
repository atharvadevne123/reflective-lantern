"""Tests for app/metrics.py."""

from __future__ import annotations

import pytest


class TestMeanAbsoluteError:
    def test_perfect_prediction(self) -> None:
        from app.metrics import mean_absolute_error

        assert mean_absolute_error([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_known_mae(self) -> None:
        from app.metrics import mean_absolute_error

        assert mean_absolute_error([0.0, 0.0], [1.0, 1.0]) == pytest.approx(1.0, abs=0.001)

    def test_empty_raises(self) -> None:
        from app.metrics import mean_absolute_error

        with pytest.raises(ValueError):
            mean_absolute_error([], [])

    def test_length_mismatch_raises(self) -> None:
        from app.metrics import mean_absolute_error

        with pytest.raises(ValueError):
            mean_absolute_error([1.0], [1.0, 2.0])


class TestRootMeanSquaredError:
    def test_perfect_prediction(self) -> None:
        from app.metrics import root_mean_squared_error

        assert root_mean_squared_error([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_known_rmse(self) -> None:
        from app.metrics import root_mean_squared_error

        result = root_mean_squared_error([0.0, 0.0], [3.0, 4.0])
        assert result == pytest.approx((9.0 + 16.0) ** 0.5 / 2**0.5, abs=0.01)

    def test_empty_raises(self) -> None:
        from app.metrics import root_mean_squared_error

        with pytest.raises(ValueError):
            root_mean_squared_error([], [])


class TestMeanAbsolutePercentageError:
    def test_perfect_prediction(self) -> None:
        from app.metrics import mean_absolute_percentage_error

        assert mean_absolute_percentage_error([10.0, 20.0], [10.0, 20.0]) == 0.0

    def test_all_zero_actual_returns_zero(self) -> None:
        from app.metrics import mean_absolute_percentage_error

        assert mean_absolute_percentage_error([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_50_percent_error(self) -> None:
        from app.metrics import mean_absolute_percentage_error

        result = mean_absolute_percentage_error([100.0], [150.0])
        assert result == pytest.approx(50.0, abs=0.01)

    def test_empty_raises(self) -> None:
        from app.metrics import mean_absolute_percentage_error

        with pytest.raises(ValueError):
            mean_absolute_percentage_error([], [])


class TestRSquared:
    def test_perfect_fit(self) -> None:
        from app.metrics import r_squared

        x = [1.0, 2.0, 3.0, 4.0]
        assert r_squared(x, x) == pytest.approx(1.0, abs=0.001)

    def test_constant_actual_returns_one(self) -> None:
        from app.metrics import r_squared

        assert r_squared([5.0, 5.0, 5.0], [5.0, 5.0, 5.0]) == 1.0

    def test_bad_fit(self) -> None:
        from app.metrics import r_squared

        actual = [1.0, 2.0, 3.0, 4.0]
        predicted = [4.0, 3.0, 2.0, 1.0]
        assert r_squared(actual, predicted) < 0

    def test_empty_raises(self) -> None:
        from app.metrics import r_squared

        with pytest.raises(ValueError):
            r_squared([], [])


class TestClassificationMetrics:
    def test_perfect_classifier(self) -> None:
        from app.metrics import classification_metrics

        result = classification_metrics([1, 0, 1, 0], [1, 0, 1, 0])
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_all_false_negatives(self) -> None:
        from app.metrics import classification_metrics

        result = classification_metrics([1, 1, 1], [0, 0, 0])
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_empty_raises(self) -> None:
        from app.metrics import classification_metrics

        with pytest.raises(ValueError):
            classification_metrics([], [])

    def test_length_mismatch_raises(self) -> None:
        from app.metrics import classification_metrics

        with pytest.raises(ValueError):
            classification_metrics([1, 0], [1])

    @pytest.mark.parametrize("tp,fp,fn,exp_f1", [(2, 1, 1, pytest.approx(2 / 3, abs=0.01))])
    def test_parametrized_f1(self, tp: int, fp: int, fn: int, exp_f1: float) -> None:
        from app.metrics import classification_metrics

        actual = [1] * tp + [0] * fp + [1] * fn
        predicted = [1] * tp + [1] * fp + [0] * fn
        result = classification_metrics(actual, predicted)
        assert result["f1"] == exp_f1
