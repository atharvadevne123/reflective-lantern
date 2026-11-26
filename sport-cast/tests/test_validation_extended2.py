"""Extended validation tests for sport-cast with parametrize and boundary coverage."""

from __future__ import annotations

import pytest

from app.validation import (
    validate_elo_range,
    validate_match_features,
    validate_probability_bounds,
    validate_win_loss_draws,
)


@pytest.mark.parametrize(
    "wins,draws,losses",
    [
        (0, 0, 0),
        (5, 0, 0),
        (0, 5, 0),
        (0, 0, 5),
        (2, 2, 1),
        (1, 2, 2),
        (3, 1, 1),
    ],
)
def test_win_loss_draws_valid_totals(wins: int, draws: int, losses: int) -> None:
    validate_win_loss_draws(wins, draws, losses)


@pytest.mark.parametrize(
    "wins,draws,losses",
    [
        (6, 0, 0),
        (2, 2, 2),
        (5, 1, 0),
        (0, 3, 3),
        (10, 0, 0),
    ],
)
def test_win_loss_draws_exceeds_five_raises(wins: int, draws: int, losses: int) -> None:
    with pytest.raises(ValueError):
        validate_win_loss_draws(wins, draws, losses)


def test_win_loss_draws_label_in_error_message() -> None:
    with pytest.raises(ValueError, match="home"):
        validate_win_loss_draws(6, 0, 0, label="home")


@pytest.mark.parametrize("valid_prob", [0.0, 0.001, 0.5, 0.999, 1.0])
def test_probability_bounds_valid(valid_prob: float) -> None:
    validate_probability_bounds(valid_prob, "test_field")


@pytest.mark.parametrize(
    "bad_prob,field",
    [
        (-0.001, "field_a"),
        (1.001, "field_b"),
        (-1.0, "field_c"),
        (2.0, "field_d"),
    ],
)
def test_probability_bounds_invalid_raises(bad_prob: float, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        validate_probability_bounds(bad_prob, field)


@pytest.mark.parametrize("valid_elo", [100.0, 500.0, 1200.0, 1500.0, 2000.0, 2500.0])
def test_elo_range_valid_values(valid_elo: float) -> None:
    validate_elo_range(valid_elo)


@pytest.mark.parametrize("bad_elo", [99.9, 0.0, -100.0, 2500.1, 3000.0])
def test_elo_range_invalid_raises(bad_elo: float) -> None:
    with pytest.raises(ValueError):
        validate_elo_range(bad_elo)


def test_elo_range_exactly_at_min() -> None:
    validate_elo_range(100.0)


def test_elo_range_exactly_at_max() -> None:
    validate_elo_range(2500.0)


def test_elo_range_name_in_error_message() -> None:
    with pytest.raises(ValueError, match="home_elo"):
        validate_elo_range(50.0, name="home_elo")


def _valid_payload(**overrides) -> dict:
    base = {
        "home_wins_last5": 2,
        "home_draws_last5": 1,
        "home_losses_last5": 2,
        "away_wins_last5": 1,
        "away_draws_last5": 2,
        "away_losses_last5": 2,
        "home_elo": 1500.0,
        "away_elo": 1500.0,
    }
    base.update(overrides)
    return base


def test_validate_match_features_valid_no_warnings() -> None:
    warnings = validate_match_features(_valid_payload())
    assert warnings == []


def test_validate_match_features_home_overflow_produces_warning() -> None:
    warnings = validate_match_features(
        _valid_payload(home_wins_last5=4, home_draws_last5=2, home_losses_last5=0)
    )
    assert any("home" in w for w in warnings)


def test_validate_match_features_away_overflow_produces_warning() -> None:
    warnings = validate_match_features(
        _valid_payload(away_wins_last5=3, away_draws_last5=2, away_losses_last5=1)
    )
    assert any("away" in w for w in warnings)


@pytest.mark.parametrize("elo_diff", [801, 1000, 1500])
def test_validate_match_features_large_elo_gap_warning(elo_diff: int) -> None:
    warnings = validate_match_features(_valid_payload(home_elo=1500.0, away_elo=1500.0 + elo_diff))
    assert any("Elo" in w for w in warnings)


def test_validate_match_features_elo_gap_exactly_800_no_warning() -> None:
    warnings = validate_match_features(_valid_payload(home_elo=1500.0, away_elo=2300.0))
    assert not any("Elo" in w for w in warnings)


def test_validate_match_features_returns_list() -> None:
    result = validate_match_features(_valid_payload())
    assert isinstance(result, list)
