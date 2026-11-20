"""Input validation utility tests."""
from __future__ import annotations

import pytest

from app.validation import (
    validate_elo_range,
    validate_match_features,
    validate_probability_bounds,
    validate_win_loss_draws,
)


def test_validate_win_loss_draws_valid():
    validate_win_loss_draws(3, 1, 1, "home")


def test_validate_win_loss_draws_exceeds_five():
    with pytest.raises(ValueError, match="exceeds 5"):
        validate_win_loss_draws(4, 2, 1, "home")


def test_validate_probability_bounds_valid():
    validate_probability_bounds(0.5, "prob")
    validate_probability_bounds(0.0, "prob")
    validate_probability_bounds(1.0, "prob")


@pytest.mark.parametrize("value", [-0.1, 1.1, 2.0])
def test_validate_probability_bounds_invalid(value):
    with pytest.raises(ValueError):
        validate_probability_bounds(value, "prob")


def test_validate_elo_range_valid():
    validate_elo_range(1500.0)
    validate_elo_range(800.0)


@pytest.mark.parametrize("elo", [0.0, 50.0, 3000.0])
def test_validate_elo_range_invalid(elo):
    with pytest.raises(ValueError):
        validate_elo_range(elo)


def test_validate_match_features_clean():
    payload = {
        "home_wins_last5": 3, "home_draws_last5": 1, "home_losses_last5": 1,
        "away_wins_last5": 2, "away_draws_last5": 1, "away_losses_last5": 2,
        "home_elo": 1600.0, "away_elo": 1500.0,
    }
    warnings = validate_match_features(payload)
    assert warnings == []


def test_validate_match_features_large_elo_gap():
    payload = {"home_elo": 2000.0, "away_elo": 1100.0}
    warnings = validate_match_features(payload)
    assert any("Elo gap" in w for w in warnings)
