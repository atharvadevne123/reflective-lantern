"""Input validation tests for Sports-Oracle."""

from __future__ import annotations

import pytest


def test_validate_probability_valid():
    from app.validation import validate_probability

    assert validate_probability(0.5, "test") == 0.5
    assert validate_probability(0.0, "test") == 0.0
    assert validate_probability(1.0, "test") == 1.0


@pytest.mark.parametrize("val", [-0.01, 1.01, 2.0])
def test_validate_probability_invalid(val):
    from app.validation import validate_probability

    with pytest.raises(ValueError):
        validate_probability(val, "test")


def test_validate_positive_valid():
    from app.validation import validate_positive

    assert validate_positive(0.5, "test") == 0.5
    assert validate_positive(100.0, "test") == 100.0


@pytest.mark.parametrize("val", [0.0, -1.0])
def test_validate_positive_invalid(val):
    from app.validation import validate_positive

    with pytest.raises(ValueError):
        validate_positive(val, "test")


def test_validate_non_negative_int_valid():
    from app.validation import validate_non_negative_int

    assert validate_non_negative_int(0, "test") == 0
    assert validate_non_negative_int(10, "test") == 10


def test_validate_non_negative_int_invalid():
    from app.validation import validate_non_negative_int

    with pytest.raises(ValueError):
        validate_non_negative_int(-1, "test")


def test_validate_match_features_valid():
    from app.validation import validate_match_features

    features = {
        "home_form": 0.7,
        "away_form": 0.4,
        "home_attack": 1.2,
        "away_attack": 1.0,
        "home_defense": 1.1,
        "away_defense": 0.9,
        "h2h_home_wins": 5,
        "h2h_draws": 3,
        "h2h_away_wins": 2,
        "home_rest_days": 7,
        "away_rest_days": 5,
    }
    result = validate_match_features(features)
    assert result["home_form"] == 0.7


def test_validate_match_features_invalid_form():
    from app.validation import validate_match_features

    with pytest.raises(ValueError):
        validate_match_features({"home_form": 1.5})
