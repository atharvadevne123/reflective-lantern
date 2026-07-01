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
