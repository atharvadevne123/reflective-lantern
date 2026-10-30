"""Tests for sqft_percentile helper."""

import pytest

from app.features import sqft_percentile


def test_empty_reference():
    assert sqft_percentile(1500.0, []) == 0.0


def test_largest_in_ref():
    ref = [1000.0, 1500.0, 2000.0]
    assert sqft_percentile(2500.0, ref) == 100.0


def test_smallest():
    ref = [1000.0, 1500.0, 2000.0]
    assert sqft_percentile(500.0, ref) == 0.0


@pytest.mark.parametrize("sqft,expected_ge", [(1001.0, 33.0), (1501.0, 66.0)])
def test_parametrized(sqft, expected_ge):
    ref = [1000.0, 1500.0, 2000.0]
    result = sqft_percentile(sqft, ref)
    assert result >= expected_ge
