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


def test_select_repo_deterministic_same_date() -> None:
    from datetime import date

    from scripts.rotate_repos import select_repo

    repos = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
    d = date(2026, 7, 9)
    r1 = select_repo(repos, d)
    r2 = select_repo(repos, d)
    assert r1 == r2


def test_select_repo_different_dates_may_differ() -> None:
    from datetime import date

    from scripts.rotate_repos import select_repo

    repos = [{"name": f"Repo{i}"} for i in range(10)]
    results = {select_repo(repos, date(2026, 7, i + 1))["name"] for i in range(10)}
    assert len(results) >= 1


def test_select_repo_raises_when_empty() -> None:
    from datetime import date

    from scripts.rotate_repos import select_repo

    with pytest.raises(ValueError, match="Need at least"):
        select_repo([], date(2026, 7, 9), min_repos=1)


def test_select_repo_returns_from_repos() -> None:
    from datetime import date

    from scripts.rotate_repos import select_repo

    repos = [{"name": "Alpha"}, {"name": "Beta"}]
    result = select_repo(repos, date(2026, 7, 9))
    assert result in repos


@pytest.mark.parametrize(
    "target_date",
    [
        date(2026, 1, 1),
        date(2026, 3, 31),
        date(2026, 6, 30),
        date(2026, 9, 1),
        date(2026, 12, 31),
    ],
)
def test_select_repo_year_boundary_dates(target_date: date) -> None:
    from scripts.rotate_repos import select_repo

    repos = [{"name": f"Repo{i}"} for i in range(5)]
    result = select_repo(repos, target_date)
    assert result in repos
    assert select_repo(repos, target_date) == result


@pytest.mark.parametrize("n_repos", [2, 5, 10, 50])
def test_select_repo_scales_to_pool_size(n_repos: int) -> None:
    from datetime import date

    from scripts.rotate_repos import select_repo

    repos = [{"name": f"Repo{i}"} for i in range(n_repos)]
    result = select_repo(repos, date(2026, 7, 11))
    assert result in repos


def test_repo_names_contains_all_valid_names() -> None:
    from scripts.rotate_repos import repo_names

    repos = [{"name": f"repo-{i}"} for i in range(10)]
    names = repo_names(repos)
    assert len(names) == 10
    assert names == sorted(names)


def test_fetch_repos_all_archived_returns_empty() -> None:
    import json
    import urllib.request
    from unittest.mock import patch

    from scripts.rotate_repos import fetch_repos

    archived_repos = [
        {"name": f"archived-{i}", "archived": True, "fork": False}
        for i in range(3)
    ]

    class FakeResponse:
        def read(self) -> bytes:
            return json.dumps(archived_repos).encode()

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    with patch.object(urllib.request, "urlopen", return_value=FakeResponse()):
        repos = fetch_repos("owner", "token")

    assert repos == []


def test_select_repo_result_is_dict() -> None:
    from datetime import date

    from scripts.rotate_repos import select_repo

    repos = [{"name": "A", "language": "Python"}, {"name": "B", "language": "Go"}]
    result = select_repo(repos, date(2026, 7, 5))
    assert isinstance(result, dict)
    assert "name" in result


def test_date_seed_is_integer() -> None:
    from datetime import date

    from scripts.rotate_repos import date_seed

    assert isinstance(date_seed(date(2026, 7, 11)), int)


def test_date_seed_different_dates_produce_different_seeds() -> None:
    from datetime import date

    from scripts.rotate_repos import date_seed

    s1 = date_seed(date(2026, 7, 1))
    s2 = date_seed(date(2026, 7, 2))
    assert s1 != s2


@pytest.mark.parametrize(
    "d,expected_seed",
    [
        (date(2026, 7, 11), 2026 * 10_000 + 7 * 100 + 11),
        (date(2026, 1, 1), 2026 * 10_000 + 1 * 100 + 1),
        (date(2026, 12, 31), 2026 * 10_000 + 12 * 100 + 31),
    ],
)
def test_date_seed_formula(d: date, expected_seed: int) -> None:
    from scripts.rotate_repos import date_seed

    assert date_seed(d) == expected_seed


def test_is_eligible_repo_filters_archived() -> None:
    from scripts.rotate_repos import _is_eligible_repo

    assert not _is_eligible_repo({"name": "repo", "archived": True, "fork": False})


def test_is_eligible_repo_filters_forks() -> None:
    from scripts.rotate_repos import _is_eligible_repo

    assert not _is_eligible_repo({"name": "repo", "archived": False, "fork": True})


def test_is_eligible_repo_filters_excluded_repo() -> None:
    from scripts.rotate_repos import EXCLUDED_REPO, _is_eligible_repo

    assert not _is_eligible_repo({"name": EXCLUDED_REPO, "archived": False, "fork": False})


def test_is_eligible_repo_passes_valid_repo() -> None:
    from scripts.rotate_repos import _is_eligible_repo

    assert _is_eligible_repo({"name": "my-repo", "archived": False, "fork": False})


def test_excluded_repo_constant_is_string() -> None:
    from scripts.rotate_repos import EXCLUDED_REPO

    assert isinstance(EXCLUDED_REPO, str)
    assert len(EXCLUDED_REPO) > 0


def test_is_eligible_repo_rejects_all_combos() -> None:
    from scripts.rotate_repos import EXCLUDED_REPO, _is_eligible_repo

    assert not _is_eligible_repo({"name": EXCLUDED_REPO, "archived": True, "fork": True})
    assert not _is_eligible_repo({"name": EXCLUDED_REPO, "archived": False, "fork": False})
    assert not _is_eligible_repo({"name": "active-repo", "archived": True, "fork": False})
    assert not _is_eligible_repo({"name": "forked-repo", "archived": False, "fork": True})


@pytest.mark.parametrize("name", ["alpha", "beta-repo", "my_project_123"])
def test_is_eligible_repo_accepts_valid_names(name: str) -> None:
    from scripts.rotate_repos import _is_eligible_repo

    assert _is_eligible_repo({"name": name, "archived": False, "fork": False})


def test_repo_names_preserves_sorted_order() -> None:
    from scripts.rotate_repos import repo_names

    repos = [{"name": "Zebra"}, {"name": "Apple"}, {"name": "Mango"}, {"name": "Banana"}]
    names = repo_names(repos)
    assert names == sorted(names)


@pytest.mark.parametrize("n", [0, 1, 5, 20])
def test_repo_names_count_matches_valid_entries(n: int) -> None:
    from scripts.rotate_repos import repo_names

    repos = [{"name": f"repo-{i}"} for i in range(n)]
    assert len(repo_names(repos)) == n
