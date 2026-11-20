"""Tests for CLI scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_train_cli_produces_artifacts(tmp_path, monkeypatch):
    """Running the train CLI writes model artifacts and exits 0."""
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "train.py"), "--samples", "200"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "lgbm_model.joblib").exists()
    assert (tmp_path / "metrics.json").exists()


def test_train_cli_help_exits_zero():
    """--help prints usage and exits 0."""
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "train.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert "--samples" in result.stdout


def test_demo_client_help_exits_zero():
    """Demo client --help works without a running server."""
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_request.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert "--host" in result.stdout
