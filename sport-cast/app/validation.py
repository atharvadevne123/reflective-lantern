"""Input validation utilities for Sport-Cast API."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def validate_win_loss_draws(wins: int, draws: int, losses: int, label: str = "") -> None:
    """Validate that win/draw/loss counts sum to at most 5 for last-5 form.

    Args:
        wins: Number of wins in last 5 matches.
        draws: Number of draws in last 5 matches.
        losses: Number of losses in last 5 matches.
        label: Optional prefix for error messages (e.g. 'home').

    Raises:
        ValueError: If total exceeds 5 matches.
    """
    total = wins + draws + losses
    if total > 5:
        raise ValueError(f"{label}: win+draw+loss={total} exceeds 5 last matches")


def validate_probability_bounds(value: float, name: str) -> None:
    """Validate a value is within [0, 1].

    Args:
        value: The probability value to validate.
        name: Field name for error messages.

    Raises:
        ValueError: If value is outside [0, 1].
    """
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name}={value} must be in [0, 1]")


def validate_elo_range(elo: float, name: str = "elo") -> None:
    """Validate Elo rating is within realistic bounds.

    Args:
        elo: Elo rating to validate.
        name: Field name for error messages.

    Raises:
        ValueError: If Elo is outside [100, 2500].
    """
    if not (100.0 <= elo <= 2500.0):
        raise ValueError(f"{name}={elo} must be between 100 and 2500")


def validate_match_features(payload: dict) -> list[str]:
    """Validate a match prediction payload and return any warnings.

    Args:
        payload: Dictionary of match feature values.

    Returns:
        List of warning strings (empty if all checks pass cleanly).
    """
    warnings: list[str] = []

    for prefix in ("home", "away"):
        try:
            validate_win_loss_draws(
                payload.get(f"{prefix}_wins_last5", 0),
                payload.get(f"{prefix}_draws_last5", 0),
                payload.get(f"{prefix}_losses_last5", 0),
                label=prefix,
            )
        except ValueError as exc:
            warnings.append(str(exc))

    home_elo = payload.get("home_elo", 1500.0)
    away_elo = payload.get("away_elo", 1500.0)
    if abs(home_elo - away_elo) > 800:
        warnings.append(f"Elo gap {abs(home_elo - away_elo):.0f} is unusually large (>800)")

    return warnings
