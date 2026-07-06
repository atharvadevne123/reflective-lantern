"""Shared pytest fixtures for the Reflective Lantern test suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generator

import pytest


@pytest.fixture()
def history_dir(tmp_path: Path) -> Path:
    """Return a temporary history directory pre-populated with sample files."""
    h = tmp_path / "history"
    h.mkdir()
    sample: list[dict[str, Any]] = [
        {
            "date": "2026-06-01",
            "mode": "improvement",
            "commits": 60,
            "tests_passed": True,
            "improvements": ["added type annotations", "added docstrings"],
        },
        {
            "date": "2026-06-15",
            "mode": "improvement",
            "commits": 60,
            "tests_passed": True,
            "improvements": ["added pytest suite"],
        },
    ]
    (h / "SampleRepo.json").write_text(json.dumps(sample))
    (h / "AnotherRepo.json").write_text(json.dumps([sample[0]]))
    return h


@pytest.fixture()
def single_entry_history_dir(tmp_path: Path) -> Path:
    """Return a history directory with a single-dict-format file."""
    h = tmp_path / "history"
    h.mkdir()
    entry = {"last_run": "2026-06-20", "commits": 45, "mode": "improvement"}
    (h / "OldRepo.json").write_text(json.dumps(entry))
    return h


@pytest.fixture()
def invalid_history_dir(tmp_path: Path) -> Path:
    """Return a history directory with an invalid JSON file."""
    h = tmp_path / "history"
    h.mkdir()
    (h / "Broken.json").write_text("{not valid json")
    return h


@pytest.fixture()
def env_with_pat(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Monkeypatch GH_PAT so scripts don't require a real token."""
    monkeypatch.setenv("GH_PAT", "test-token-123")
    monkeypatch.setenv("GITHUB_USERNAME", "test-owner")
    yield
    # teardown handled by monkeypatch


@pytest.fixture()
def settings_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Set all required env vars for Settings."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("GH_PAT", "ghp_test")
    monkeypatch.setenv("NOTION_API_KEY", "secret_test")
    monkeypatch.setenv("GMAIL_USER", "test@gmail.com")
    monkeypatch.setenv("GMAIL_APP_PASS", "test-pass")
    yield


@pytest.fixture()
def multi_repo_history_dir(tmp_path: Path) -> Path:
    """Return a history directory with multiple repos across different modes."""
    h = tmp_path / "history"
    h.mkdir()
    (h / "Alpha.json").write_text(json.dumps([
        {"date": "2026-07-01", "mode": "improvement", "commits": 60, "tests_passed": True},
        {"date": "2026-07-03", "mode": "improvement", "commits": 60, "tests_passed": True},
    ]))
    (h / "Beta.json").write_text(json.dumps([
        {"date": "2026-07-08", "mode": "innovation", "commits": 120, "tests_passed": True},
    ]))
    (h / "Gamma.json").write_text(json.dumps([
        {"date": "2026-07-02", "mode": "improvement", "commits": 60, "tests_passed": False},
    ]))
    return h


@pytest.fixture()
def innovation_history_dir(tmp_path: Path) -> Path:
    """Return a history directory with an INNOVATION mode entry."""
    h = tmp_path / "history"
    h.mkdir()
    (h / "NewProject.json").write_text(json.dumps([{
        "date": "2026-07-08",
        "mode": "innovation",
        "commits": 114,
        "tests_passed": True,
        "improvements": ["built new ML pipeline from scratch"],
    }]))
    return h
