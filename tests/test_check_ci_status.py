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


@pytest.mark.parametrize("conclusion,is_fail", [
    ("success", False),
    ("failure", True),
    ("timed_out", True),
    ("skipped", False),
    ("cancelled", False),
])
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


@pytest.mark.parametrize("runs,expected_count", [
    ([], 0),
    ([{"workflow_id": 1, "name": "CI", "conclusion": "success"}], 1),
    (
        [
            {"workflow_id": 1, "name": "CI", "conclusion": "success"},
            {"workflow_id": 2, "name": "Deploy", "conclusion": "failure"},
        ],
        2,
    ),
])
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
