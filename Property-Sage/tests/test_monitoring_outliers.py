"""Tests for detect_outlier_predictions."""

import pytest

from app.monitoring import detect_outlier_predictions


def test_no_outliers():
    prices = [100_000.0, 110_000.0, 105_000.0, 108_000.0]
    assert detect_outlier_predictions(prices) == []


def test_with_outlier():
    prices = [100_000.0] * 20 + [10_000_000.0]
    outliers = detect_outlier_predictions(prices)
    assert 20 in outliers


def test_too_few_samples():
    assert detect_outlier_predictions([1.0, 2.0]) == []


@pytest.mark.parametrize("threshold,expect_outlier", [(1.0, True), (10.0, False)])
def test_threshold_effect(threshold, expect_outlier):
    prices = [100_000.0] * 10 + [500_000.0]
    outliers = detect_outlier_predictions(prices, z_threshold=threshold)
    assert (len(outliers) > 0) == expect_outlier
