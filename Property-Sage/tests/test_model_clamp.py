"""Tests for clamp_prediction."""

import pytest

from app.model import clamp_prediction


def test_normal_value():
    assert clamp_prediction(500_000.0) == 500_000.0


def test_too_low():
    assert clamp_prediction(100.0) == 10_000.0


def test_too_high():
    assert clamp_prediction(100_000_000.0) == 50_000_000.0


@pytest.mark.parametrize("val,expected", [
    (5000.0, 10_000.0),
    (1_000_000.0, 1_000_000.0),
])
def test_clamp_parametrized(val, expected):
    assert clamp_prediction(val) == expected
