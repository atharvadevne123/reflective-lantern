"""Tests for scripts.rotate_repos."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest


SAMPLE_REPOS: list[dict[str, Any]] = [
    {"name": "Alpha", "language": "Python", "archived": False, "fork": False},
    {"name": "Beta", "language": "JavaScript", "archived": False, "fork": False},
    {"name": "Gamma", "language": "Python", "archived": False, "fork": False},
    {"name": "reflective-lantern", "language": "Python", "archived": False, "fork": False},
]


def test_select_repo_is_deterministic() -> None:
    from scripts.rotate_repos import select_repo
    r1 = select_repo(SAMPLE_REPOS[:-1], date(2026, 7, 1))
    r2 = select_repo(SAMPLE_REPOS[:-1], date(2026, 7, 1))
    assert r1["name"] == r2["name"]


def test_select_repo_varies_by_date() -> None:
    from scripts.rotate_repos import select_repo
    repos = SAMPLE_REPOS[:-1]  # exclude reflective-lantern
    results = {select_repo(repos, date(2026, 7, d))["name"] for d in range(1, 10)}
    # With 3 repos over 9 days, should see at least 2 distinct names
    assert len(results) >= 2


def test_select_repo_picks_from_given_list() -> None:
    from scripts.rotate_repos import select_repo
    repos = [{"name": "OnlyOne"}]
    result = select_repo(repos, date(2026, 7, 1))
    assert result["name"] == "OnlyOne"


@pytest.mark.parametrize("target_date", [
    date(2026, 7, 1),
    date(2026, 7, 7),
    date(2026, 12, 31),
])
def test_select_repo_always_in_list(target_date: date) -> None:
    from scripts.rotate_repos import select_repo
    repos = SAMPLE_REPOS[:-1]
    result = select_repo(repos, target_date)
    assert result in repos


def test_select_repo_different_seeds_different_results() -> None:
    from scripts.rotate_repos import select_repo
    repos = [{"name": f"Repo{i}"} for i in range(20)]
    dates = [date(2026, 1, d) for d in range(1, 20)]
    names = [select_repo(repos, d)["name"] for d in dates]
    # Should not all be the same repo over 19 days
    assert len(set(names)) > 1
