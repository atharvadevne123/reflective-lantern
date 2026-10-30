"""Tests for calculate_confidence_score helper."""

import pytest

from app.model import calculate_confidence_score


def test_confidence_empty_cv():
    assert calculate_confidence_score([], 100.0, 100.0, 10.0) == 0.0


def test_confidence_perfect_cv():
    score = calculate_confidence_score([1.0, 1.0, 1.0], 100.0, 100.0, 10.0)
    assert 0.9 <= score <= 1.0


def test_confidence_far_from_mean():
    score = calculate_confidence_score([0.9, 0.9], 10_000.0, 100.0, 10.0)
    assert score < 0.9


def test_confidence_clipped_to_zero():
    score = calculate_confidence_score([0.1], 100_000.0, 100.0, 1.0)
    assert score >= 0.0


@pytest.mark.parametrize("r2,expected_ge", [(0.8, 0.5), (0.5, 0.0)])
def test_confidence_parametrized(r2, expected_ge):
    score = calculate_confidence_score([r2], 100.0, 100.0, 10.0)
    assert score >= expected_ge
