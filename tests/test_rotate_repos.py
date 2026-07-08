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


@pytest.mark.parametrize(
    "target_date",
    [
        date(2026, 7, 1),
        date(2026, 7, 7),
        date(2026, 12, 31),
    ],
)
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


def test_fetch_repos_filters_archived() -> None:
    """fetch_repos should exclude archived repos."""
    import json
    import urllib.request
    from unittest.mock import patch

    from scripts.rotate_repos import fetch_repos

    mock_repos = [
        {"name": "Active", "archived": False, "fork": False},
        {"name": "Archived", "archived": True, "fork": False},
        {"name": "Fork", "archived": False, "fork": True},
        {"name": "reflective-lantern", "archived": False, "fork": False},
    ]

    class FakeResponse:
        def read(self) -> bytes:
            return json.dumps(mock_repos).encode()

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    with patch.object(urllib.request, "urlopen", return_value=FakeResponse()):
        repos = fetch_repos("owner", "token")

    names = [r["name"] for r in repos]
    assert "Active" in names
    assert "Archived" not in names
    assert "Fork" not in names
    assert "reflective-lantern" not in names


@pytest.mark.parametrize(
    "seed_date",
    [
        date(2026, 1, 1),
        date(2026, 6, 15),
        date(2026, 12, 31),
    ],
)
def test_select_repo_seed_is_date_based(seed_date: date) -> None:
    from scripts.rotate_repos import select_repo

    repos = [{"name": f"R{i}"} for i in range(10)]
    result = select_repo(repos, seed_date)
    assert result in repos
    # Deterministic: same date should give same result
    assert select_repo(repos, seed_date)["name"] == result["name"]


def test_select_repo_raises_on_empty_list() -> None:
    from datetime import date

    from scripts.rotate_repos import select_repo

    with pytest.raises(ValueError, match="at least 1"):
        select_repo([], date(2026, 7, 3))


def test_select_repo_single_item_always_returns_it() -> None:
    from datetime import date

    from scripts.rotate_repos import select_repo

    repo = {"name": "only-repo", "language": "Python"}
    for day in range(1, 32):
        try:
            d = date(2026, 7, day)
        except ValueError:
            continue
        assert select_repo([repo], d) is repo


def test_select_repo_excludes_nothing_from_single_list() -> None:
    from datetime import date

    from scripts.rotate_repos import select_repo

    repos = [{"name": f"repo-{i}"} for i in range(10)]
    result = select_repo(repos, date(2026, 7, 3))
    assert result in repos


def test_select_repo_min_repos_raises_when_too_few() -> None:
    from datetime import date

    from scripts.rotate_repos import select_repo

    repos = [{"name": "solo"}]
    with pytest.raises(ValueError, match="at least 2"):
        select_repo(repos, date(2026, 7, 6), min_repos=2)


def test_select_repo_min_repos_passes_when_enough() -> None:
    from datetime import date

    from scripts.rotate_repos import select_repo

    repos = [{"name": "a"}, {"name": "b"}]
    result = select_repo(repos, date(2026, 7, 6), min_repos=2)
    assert result in repos


def test_select_repo_min_repos_default_allows_single() -> None:
    from datetime import date

    from scripts.rotate_repos import select_repo

    result = select_repo([{"name": "solo"}], date(2026, 7, 6))
    assert result["name"] == "solo"


def test_repo_names_sorted() -> None:
    from scripts.rotate_repos import repo_names

    repos = [{"name": "Zebra"}, {"name": "Apple"}, {"name": "Mango"}]
    assert repo_names(repos) == ["Apple", "Mango", "Zebra"]


def test_repo_names_empty() -> None:
    from scripts.rotate_repos import repo_names

    assert repo_names([]) == []


def test_repo_names_skips_non_string_names() -> None:
    from scripts.rotate_repos import repo_names

    repos = [{"name": "Valid"}, {"name": None}, {"other": "no-name"}]
    result = repo_names(repos)
    assert result == ["Valid"]
