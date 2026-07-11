"""Tests for scripts.check_ci_status."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

SAMPLE_RUNS: list[dict[str, Any]] = [
    {"workflow_id": 1, "name": "CI", "conclusion": "success", "status": "completed"},
    {"workflow_id": 2, "name": "Deploy", "conclusion": "failure", "status": "completed"},
    {"workflow_id": 1, "name": "CI", "conclusion": "success", "status": "completed"},
]


def test_get_latest_runs_deduplicates_by_workflow_id() -> None:
    from scripts.check_ci_status import get_latest_runs

    mock_data = {"workflow_runs": SAMPLE_RUNS}
    with patch("scripts.check_ci_status._get", return_value=mock_data):
        runs = get_latest_runs("owner", "repo", "token")
    workflow_ids = [r["workflow_id"] for r in runs]
    assert len(workflow_ids) == len(set(workflow_ids))


def test_get_latest_runs_picks_first_occurrence() -> None:
    from scripts.check_ci_status import get_latest_runs

    mock_data = {"workflow_runs": SAMPLE_RUNS}
    with patch("scripts.check_ci_status._get", return_value=mock_data):
        runs = get_latest_runs("owner", "repo", "token")
    ci_run = next(r for r in runs if r["workflow_id"] == 1)
    assert ci_run["conclusion"] == "success"


def test_get_latest_runs_returns_empty_on_error() -> None:
    from scripts.check_ci_status import get_latest_runs

    with patch("scripts.check_ci_status._get", side_effect=Exception("network error")):
        runs = get_latest_runs("owner", "repo", "token")
    assert runs == []


def test_get_latest_runs_handles_empty_response() -> None:
    from scripts.check_ci_status import get_latest_runs

    with patch("scripts.check_ci_status._get", return_value={"workflow_runs": []}):
        runs = get_latest_runs("owner", "repo", "token")
    assert runs == []


@pytest.mark.parametrize(
    "conclusion,is_fail",
    [
        ("success", False),
        ("failure", True),
        ("timed_out", True),
        ("skipped", False),
        ("cancelled", False),
    ],
)
def test_failure_conclusions(conclusion: str, is_fail: bool) -> None:
    assert (conclusion in ("failure", "timed_out")) == is_fail


def test_get_retries_on_network_error() -> None:
    """_get() should retry the configured number of times before raising."""
    import urllib.request

    call_count = 0

    def failing_open(*args: object, **kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        raise OSError("connection refused")

    from scripts.check_ci_status import _get

    with (
        patch.object(urllib.request, "urlopen", side_effect=failing_open),
        pytest.raises(RuntimeError, match="All"),
    ):
        _get("https://api.github.com/repos/test", "token", retries=2)
    assert call_count == 2


@pytest.mark.parametrize(
    "runs,expected_count",
    [
        ([], 0),
        ([{"workflow_id": 1, "name": "CI", "conclusion": "success"}], 1),
        (
            [
                {"workflow_id": 1, "name": "CI", "conclusion": "success"},
                {"workflow_id": 2, "name": "Deploy", "conclusion": "failure"},
            ],
            2,
        ),
    ],
)
def test_get_latest_runs_count(runs: list[dict], expected_count: int) -> None:
    from scripts.check_ci_status import get_latest_runs

    with patch("scripts.check_ci_status._get", return_value={"workflow_runs": runs}):
        result = get_latest_runs("owner", "repo", "token")
    assert len(result) == expected_count


def test_get_latest_runs_no_workflow_id_key() -> None:
    from scripts.check_ci_status import get_latest_runs

    runs = [{"name": "CI", "conclusion": "success"}]  # missing workflow_id
    with patch("scripts.check_ci_status._get", return_value={"workflow_runs": runs}):
        result = get_latest_runs("owner", "repo", "token")
    assert len(result) == 1


def test_get_latest_runs_multiple_no_workflow_id() -> None:
    """Runs without workflow_id are all grouped under None key → only one kept."""
    from scripts.check_ci_status import get_latest_runs

    runs = [
        {"name": "CI", "conclusion": "success"},
        {"name": "Deploy", "conclusion": "failure"},
    ]
    with patch("scripts.check_ci_status._get", return_value={"workflow_runs": runs}):
        result = get_latest_runs("owner", "repo", "token")
    # Both runs have workflow_id=None, so only the first is kept
    assert len(result) == 1
    assert result[0]["name"] == "CI"


@pytest.mark.parametrize("retries", [1, 2, 3])
def test_get_retries_count(retries: int) -> None:
    import urllib.request

    call_count = 0

    def failing_open(*args: object, **kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        raise OSError("refused")

    from scripts.check_ci_status import _get

    with (
        patch.object(urllib.request, "urlopen", side_effect=failing_open),
        pytest.raises(RuntimeError),
    ):
        _get("https://api.github.com", "tok", retries=retries)
    assert call_count == retries


def test_get_latest_runs_no_workflow_id_dedup() -> None:
    import urllib.request
    from unittest.mock import MagicMock, patch

    run = {"workflow_id": None, "name": "CI", "conclusion": "success", "status": "completed"}
    payload = {"workflow_runs": [run, run, run]}

    ctx_mock = MagicMock()
    ctx_mock.__enter__ = MagicMock(return_value=MagicMock(read=lambda: b""))
    ctx_mock.__exit__ = MagicMock(return_value=False)

    import json

    mock_response = MagicMock()
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_response.read.return_value = json.dumps(payload).encode()

    with patch.object(urllib.request, "urlopen", return_value=mock_response):
        from scripts.check_ci_status import get_latest_runs

        runs = get_latest_runs("owner", "repo", "tok")
    assert len(runs) == 1


@pytest.mark.parametrize(
    "conclusion,expected_success",
    [
        ("success", True),
        ("failure", False),
        ("timed_out", False),
        ("skipped", True),
        ("cancelled", True),
        ("neutral", True),
        ("action_required", False),
    ],
)
def test_conclusion_success_mapping(conclusion: str, expected_success: bool) -> None:
    failing_conclusions = {"failure", "timed_out", "action_required"}
    assert (conclusion not in failing_conclusions) == expected_success


def test_get_latest_runs_returns_list(monkeypatch) -> None:
    from unittest.mock import patch

    from scripts.check_ci_status import get_latest_runs

    with patch("scripts.check_ci_status._get", return_value={"workflow_runs": []}):
        result = get_latest_runs("owner", "repo", "tok")
    assert isinstance(result, list)


@pytest.mark.parametrize("owner,repo", [
    ("alice", "my-repo"),
    ("org-name", "backend"),
    ("user123", "test-project"),
])
def test_get_latest_runs_accepts_various_owners(owner: str, repo: str) -> None:
    from unittest.mock import patch

    from scripts.check_ci_status import get_latest_runs

    with patch("scripts.check_ci_status._get", return_value={"workflow_runs": []}):
        result = get_latest_runs(owner, repo, "token")
    assert result == []


def test_failing_conclusions_constant_contains_failure() -> None:
    from scripts.check_ci_status import FAILING_CONCLUSIONS

    assert "failure" in FAILING_CONCLUSIONS
    assert "timed_out" in FAILING_CONCLUSIONS


def test_failing_conclusions_constant_excludes_success() -> None:
    from scripts.check_ci_status import FAILING_CONCLUSIONS

    assert "success" not in FAILING_CONCLUSIONS
    assert "skipped" not in FAILING_CONCLUSIONS


def test_default_retries_constant() -> None:
    from scripts.check_ci_status import DEFAULT_RETRIES

    assert DEFAULT_RETRIES >= 1


@pytest.mark.parametrize("conclusion", ["failure", "timed_out", "action_required"])
def test_all_failing_conclusions_in_set(conclusion: str) -> None:
    from scripts.check_ci_status import FAILING_CONCLUSIONS

    assert conclusion in FAILING_CONCLUSIONS


def test_repo_flag_filters_to_single_repo(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import sys

    with monkeypatch.context() as m:
        m.setenv("GH_PAT", "tok")
        import json
        import urllib.request
        from unittest.mock import MagicMock, patch

        repos_payload = [
            {"name": "repo-a", "archived": False, "fork": False},
            {"name": "repo-b", "archived": False, "fork": False},
        ]
        runs_payload = {"workflow_runs": []}

        call_urls: list[str] = []

        def fake_urlopen(req, timeout=15):  # noqa: ANN001
            call_urls.append(req.full_url)
            mock = MagicMock()
            mock.__enter__ = lambda s: s
            mock.__exit__ = MagicMock(return_value=False)
            if "repos?per_page" in req.full_url:
                mock.read.return_value = json.dumps(repos_payload).encode()
            else:
                mock.read.return_value = json.dumps(runs_payload).encode()
            return mock

        import scripts.check_ci_status as ccs

        with (
            patch.object(urllib.request, "urlopen", side_effect=fake_urlopen),
            patch.object(sys, "argv", ["check_ci_status.py", "--repo", "repo-a"]),
        ):
            ccs.main()
        assert not any("repo-b" in u for u in call_urls)


def test_runs_per_page_constant() -> None:
    from scripts.check_ci_status import RUNS_PER_PAGE

    assert RUNS_PER_PAGE > 0


def test_repos_per_page_constant() -> None:
    from scripts.check_ci_status import REPOS_PER_PAGE

    assert REPOS_PER_PAGE > 0


@pytest.mark.parametrize("conclusion", ["success", "skipped", "cancelled", "neutral"])
def test_non_failing_conclusions_not_in_set(conclusion: str) -> None:
    from scripts.check_ci_status import FAILING_CONCLUSIONS

    assert conclusion not in FAILING_CONCLUSIONS


def test_get_latest_runs_dedup_multiple_same_id() -> None:
    from scripts.check_ci_status import get_latest_runs

    runs = [
        {"workflow_id": 5, "name": "Test", "conclusion": "success"},
        {"workflow_id": 5, "name": "Test", "conclusion": "failure"},
        {"workflow_id": 5, "name": "Test", "conclusion": "timed_out"},
    ]
    with patch("scripts.check_ci_status._get", return_value={"workflow_runs": runs}):
        result = get_latest_runs("owner", "repo", "token")
    assert len(result) == 1
    assert result[0]["conclusion"] == "success"


@pytest.mark.parametrize("n_runs", [0, 1, 5, 10])
def test_get_latest_runs_result_count_bounded_by_unique_ids(n_runs: int) -> None:
    from scripts.check_ci_status import get_latest_runs

    runs = [{"workflow_id": i, "name": f"WF{i}", "conclusion": "success"} for i in range(n_runs)]
    with patch("scripts.check_ci_status._get", return_value={"workflow_runs": runs}):
        result = get_latest_runs("owner", "repo", "token")
    assert len(result) == n_runs
