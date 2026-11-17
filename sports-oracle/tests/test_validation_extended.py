"""Extended tests for Sports-Oracle input validation utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.validation import (
    validate_match_features,
    validate_non_negative_int,
    validate_positive,
    validate_probability,
)


def test_validate_probability_zero():
    assert validate_probability(0.0, "p") == 0.0


def test_validate_probability_one():
    assert validate_probability(1.0, "p") == 1.0


def test_validate_probability_midpoint():
    assert validate_probability(0.5, "p") == 0.5


def test_validate_probability_negative_raises():
    with pytest.raises(ValueError):
        validate_probability(-0.1, "p")


def test_validate_probability_above_one_raises():
    with pytest.raises(ValueError):
        validate_probability(1.1, "p")


def test_validate_positive_returns_float():
    result = validate_positive(2.5, "attack")
    assert isinstance(result, float)
    assert result == 2.5


def test_validate_positive_zero_raises():
    with pytest.raises(ValueError):
        validate_positive(0.0, "val")


def test_validate_positive_negative_raises():
    with pytest.raises(ValueError):
        validate_positive(-1.0, "val")


def test_validate_non_negative_int_zero():
    assert validate_non_negative_int(0, "wins") == 0


def test_validate_non_negative_int_positive():
    assert validate_non_negative_int(5, "wins") == 5


def test_validate_non_negative_int_negative_raises():
    with pytest.raises(ValueError):
        validate_non_negative_int(-1, "wins")


def test_validate_match_features_passes_through_extra_fields():
    features = {"home_form": 0.6, "away_form": 0.4, "extra_field": "data"}
    result = validate_match_features(features)
    assert "extra_field" in result


def test_validate_match_features_validates_probabilities():
    with pytest.raises(ValueError):
        validate_match_features({"home_form": 1.5})


def test_validate_match_features_validates_attack_positive():
    with pytest.raises(ValueError):
        validate_match_features({"home_attack": -1.0})


def test_validate_match_features_validates_h2h_non_negative():
    with pytest.raises(ValueError):
        validate_match_features({"h2h_home_wins": -1})


def test_validate_match_features_rest_days_valid():
    result = validate_match_features({"home_rest_days": 7})
    assert result["home_rest_days"] == 7


def test_validate_match_features_rest_days_too_high_raises():
    with pytest.raises(ValueError):
        validate_match_features({"home_rest_days": 31})


def test_validate_match_features_empty_dict():
    result = validate_match_features({})
    assert isinstance(result, dict)


@pytest.mark.parametrize("prob", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_validate_probability_valid_range(prob):
    assert validate_probability(prob, "test") == prob
