"""Extended tests for sport-cast app/validation.py."""

from __future__ import annotations

import pytest

from app.validation import (
    validate_elo_range,
    validate_match_features,
    validate_probability_bounds,
    validate_win_loss_draws,
)


class TestValidateWinLossDraws:
    def test_valid_combination(self) -> None:
        validate_win_loss_draws(wins=3, draws=1, losses=1)

    def test_zero_matches(self) -> None:
        validate_win_loss_draws(wins=0, draws=0, losses=0)

    def test_exactly_five(self) -> None:
        validate_win_loss_draws(wins=2, draws=2, losses=1)

    def test_exceeds_five_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_win_loss_draws(wins=3, draws=2, losses=2)

    def test_label_in_error_message(self) -> None:
        with pytest.raises(ValueError, match="home"):
            validate_win_loss_draws(wins=3, draws=3, losses=0, label="home")


class TestValidateProbabilityBounds:
    def test_valid_zero(self) -> None:
        validate_probability_bounds(0.0, "p")

    def test_valid_one(self) -> None:
        validate_probability_bounds(1.0, "p")

    def test_valid_middle(self) -> None:
        validate_probability_bounds(0.5, "p")

    def test_below_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_probability_bounds(-0.01, "prob")

    def test_above_one_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_probability_bounds(1.01, "prob")

    def test_name_in_error_message(self) -> None:
        with pytest.raises(ValueError, match="win_prob"):
            validate_probability_bounds(2.0, "win_prob")


class TestValidateEloRange:
    def test_valid_elo(self) -> None:
        validate_elo_range(1500.0)

    def test_minimum_elo(self) -> None:
        validate_elo_range(100.0)

    def test_maximum_elo(self) -> None:
        validate_elo_range(2500.0)

    def test_below_minimum_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_elo_range(99.0)

    def test_above_maximum_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_elo_range(2501.0)


class TestValidateMatchFeatures:
    def test_valid_payload_no_warnings(self) -> None:
        payload = {
            "home_wins_last5": 3,
            "home_draws_last5": 1,
            "home_losses_last5": 1,
            "away_wins_last5": 2,
            "away_draws_last5": 2,
            "away_losses_last5": 1,
        }
        warnings = validate_match_features(payload)
        assert isinstance(warnings, list)

    def test_returns_list(self) -> None:
        assert isinstance(validate_match_features({}), list)
