"""Tests for scripts.health_check."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from scripts.health_check import RepoHealth


def test_repo_health_healthy_by_default() -> None:
    h = RepoHealth(name="Repo")
    assert h.healthy is True


def test_repo_health_unhealthy_with_failing_workflows() -> None:
    h = RepoHealth(name="Repo", failing_workflows=["CI"])
    assert h.healthy is False


def test_repo_health_unhealthy_with_open_branches() -> None:
    h = RepoHealth(name="Repo", open_branches=["feature/x"])
    assert h.healthy is False


def test_repo_health_unhealthy_without_release() -> None:
    h = RepoHealth(name="Repo", has_release=False)
    assert h.healthy is False


def test_repo_health_unhealthy_without_ci() -> None:
    h = RepoHealth(name="Repo", has_ci=False)
    assert h.healthy is False


def test_repo_health_all_issues() -> None:
    h = RepoHealth(
        name="Repo",
        failing_workflows=["CI", "Deploy"],
        open_branches=["dev"],
        has_release=False,
        has_ci=False,
    )
    assert h.healthy is False


def test_check_repo_marks_failing_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.health_check import check_repo

    def mock_get(url: str, token: str) -> Any:
        if "actions/runs" in url:
            return {"workflow_runs": [{"workflow_id": 1, "name": "CI", "conclusion": "failure"}]}
        if "branches" in url:
            return [{"name": "main"}]
        if "releases" in url:
            return [{"tag_name": "v1.0.0"}]
        return {}

    with patch("scripts.health_check._get", side_effect=mock_get):
        health = check_repo("my-repo", "main", "token")

    assert "CI" in health.failing_workflows


def test_check_repo_detects_open_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.health_check import check_repo

    def mock_get(url: str, token: str) -> Any:
        if "actions/runs" in url:
            return {"workflow_runs": []}
        if "branches" in url:
            return [{"name": "main"}, {"name": "feature/new"}]
        if "releases" in url:
            return [{"tag_name": "v1.0.0"}]
        return {}

    with patch("scripts.health_check._get", side_effect=mock_get):
        health = check_repo("my-repo", "main", "token")

    assert "feature/new" in health.open_branches


@pytest.mark.parametrize(
    "workflows,expected_failing",
    [
        ([], []),
        ([{"workflow_id": 1, "name": "CI", "conclusion": "success"}], []),
        ([{"workflow_id": 1, "name": "CI", "conclusion": "failure"}], ["CI"]),
        ([{"workflow_id": 1, "name": "CI", "conclusion": "timed_out"}], ["CI"]),
    ],
)
def test_check_repo_workflow_conclusions(
    workflows: list[dict[str, str]], expected_failing: list[str]
) -> None:
    from scripts.health_check import check_repo

    def mock_get(url: str, token: str) -> Any:
        if "actions/runs" in url:
            return {"workflow_runs": workflows}
        if "branches" in url:
            return [{"name": "main"}]
        if "releases" in url:
            return [{"tag_name": "v1.0.0"}]
        return {}

    with patch("scripts.health_check._get", side_effect=mock_get):
        health = check_repo("repo", "main", "token")

    assert health.failing_workflows == expected_failing


def test_get_retries_on_failure() -> None:
    """_get() should retry the configured number of times before raising."""
    from scripts.health_check import _get

    call_count = 0

    def failing_urlopen(*args: object, **kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        raise OSError("network error")

    import urllib.request

    with (
        patch.object(urllib.request, "urlopen", side_effect=failing_urlopen),
        pytest.raises(RuntimeError, match="All"),
    ):
        _get("https://api.github.com/test", "token", retries=2)
    assert call_count == 2


def test_repo_health_name_stored() -> None:
    h = RepoHealth(name="My-Repo")
    assert h.name == "My-Repo"


@pytest.mark.parametrize(
    "has_release,has_ci,branches,workflows,healthy",
    [
        (True, True, [], [], True),
        (False, True, [], [], False),
        (True, False, [], [], False),
        (True, True, ["feat/x"], [], False),
        (True, True, [], ["CI"], False),
    ],
)
def test_repo_health_parametrized(
    has_release: bool,
    has_ci: bool,
    branches: list[str],
    workflows: list[str],
    healthy: bool,
) -> None:
    h = RepoHealth(
        name="R",
        has_release=has_release,
        has_ci=has_ci,
        open_branches=branches,
        failing_workflows=workflows,
    )
    assert h.healthy is healthy


def test_repo_health_stats_healthy() -> None:
    from scripts.health_check import RepoHealth

    r = RepoHealth(name="clean-repo")
    stats = r.stats()
    assert stats["healthy"] is True
    assert stats["failing_workflows"] == 0
    assert stats["open_branches"] == 0
    assert stats["has_release"] is True
    assert stats["has_ci"] is True


def test_repo_health_stats_with_issues() -> None:
    from scripts.health_check import RepoHealth

    r = RepoHealth(
        name="broken",
        failing_workflows=["CI", "Lint"],
        open_branches=["feature/x"],
        has_release=False,
    )
    stats = r.stats()
    assert stats["healthy"] is False
    assert stats["failing_workflows"] == 2
    assert stats["open_branches"] == 1
    assert stats["has_release"] is False


@pytest.mark.parametrize(
    "failing,branches,has_release,has_ci,expected_healthy",
    [
        ([], [], True, True, True),
        (["CI"], [], True, True, False),
        ([], ["branch-x"], True, True, False),
        ([], [], False, True, False),
        ([], [], True, False, False),
    ],
)
def test_repo_health_healthy_parametrized(
    failing: list, branches: list, has_release: bool, has_ci: bool, expected_healthy: bool
) -> None:
    from scripts.health_check import RepoHealth

    r = RepoHealth(
        name="repo",
        failing_workflows=failing,
        open_branches=branches,
        has_release=has_release,
        has_ci=has_ci,
    )
    assert r.healthy is expected_healthy


def test_issues_count_all_healthy() -> None:
    h = RepoHealth(name="Repo")
    assert h.issues_count == 0


def test_issues_count_with_failing_workflows() -> None:
    h = RepoHealth(name="Repo", failing_workflows=["CI", "lint"])
    assert h.issues_count == 2


def test_issues_count_with_all_issues() -> None:
    h = RepoHealth(
        name="Repo",
        failing_workflows=["CI"],
        open_branches=["dev", "hotfix"],
        has_release=False,
        has_ci=False,
    )
    assert h.issues_count == 5


def test_stats_includes_issues_count() -> None:
    h = RepoHealth(name="Repo", failing_workflows=["CI"])
    s = h.stats()
    assert "issues_count" in s
    assert s["issues_count"] == 1


def test_summary_line_healthy() -> None:
    h = RepoHealth(name="MyRepo")
    assert h.summary_line() == "MyRepo: OK"


def test_summary_line_with_issues() -> None:
    h = RepoHealth(name="BadRepo", failing_workflows=["CI"])
    assert "ISSUES" in h.summary_line()
    assert "1" in h.summary_line()


def test_summary_line_contains_repo_name() -> None:
    h = RepoHealth(name="SpecialRepo", open_branches=["dev"])
    line = h.summary_line()
    assert "SpecialRepo" in line


def test_stats_has_all_expected_keys() -> None:
    h = RepoHealth(name="Repo")
    s = h.stats()
    for key in ("failing_workflows", "open_branches", "has_release", "has_ci", "healthy"):
        assert key in s


def test_issues_count_zero_when_healthy() -> None:
    h = RepoHealth(name="Repo")
    assert h.issues_count == 0


def test_issues_count_counts_each_failing_workflow_separately() -> None:
    h = RepoHealth(name="Repo", failing_workflows=["CI", "Lint", "Deploy"])
    assert h.issues_count == 3


def test_stats_has_release_reflects_field() -> None:
    h = RepoHealth(name="Repo", has_release=False)
    assert h.stats()["has_release"] is False


def test_repo_health_name_preserved() -> None:
    h = RepoHealth(name="my-repo-123")
    assert h.name == "my-repo-123"


def test_failing_conclusions_constant_contains_failure() -> None:
    from scripts.health_check import FAILING_CONCLUSIONS

    assert "failure" in FAILING_CONCLUSIONS
    assert "timed_out" in FAILING_CONCLUSIONS


def test_failing_conclusions_constant_excludes_success() -> None:
    from scripts.health_check import FAILING_CONCLUSIONS

    assert "success" not in FAILING_CONCLUSIONS


def test_default_retries_constant_is_positive() -> None:
    from scripts.health_check import DEFAULT_RETRIES

    assert DEFAULT_RETRIES >= 1


@pytest.mark.parametrize("conclusion", ["failure", "timed_out"])
def test_failing_conclusions_contains_expected(conclusion: str) -> None:
    from scripts.health_check import FAILING_CONCLUSIONS

    assert conclusion in FAILING_CONCLUSIONS


def test_workflow_runs_per_page_constant() -> None:
    from scripts.health_check import WORKFLOW_RUNS_PER_PAGE

    assert WORKFLOW_RUNS_PER_PAGE > 0


def test_branches_per_page_constant() -> None:
    from scripts.health_check import BRANCHES_PER_PAGE

    assert BRANCHES_PER_PAGE > 0


def test_releases_per_page_constant() -> None:
    from scripts.health_check import RELEASES_PER_PAGE

    assert RELEASES_PER_PAGE >= 1
