"""Tests for the standalone CLI scripts.

These invoke the scripts as subprocesses rather than importing them: the failure
mode they guard against — running a script directly puts ``scripts/`` on
sys.path instead of the project root, so ``app`` is not importable — is
invisible to an in-process import.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP = PROJECT_ROOT / "scripts" / "bootstrap_reference.py"
DIAGRAM = PROJECT_ROOT / "scripts" / "generate_diagram.py"


def _run(script: Path, *args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=300,
    )


class TestBootstrapReferenceScript:
    def test_runs_from_project_root(self, tmp_path: Path) -> None:
        out = tmp_path / "ref.json"
        result = _run(BOOTSTRAP, "--samples", "50", "--out", str(out), cwd=PROJECT_ROOT)
        assert result.returncode == 0, result.stderr
        assert out.exists()

    def test_runs_from_an_unrelated_cwd(self, tmp_path: Path) -> None:
        out = tmp_path / "ref.json"
        result = _run(BOOTSTRAP, "--samples", "50", "--out", str(out), cwd=tmp_path)
        assert result.returncode == 0, result.stderr

    def test_output_is_valid_per_feature_json(self, tmp_path: Path) -> None:
        out = tmp_path / "ref.json"
        _run(BOOTSTRAP, "--samples", "60", "--out", str(out), cwd=PROJECT_ROOT)
        data = json.loads(out.read_text())
        assert "depth_km" in data
        assert "prediction" in data
        assert len(data["depth_km"]) == 60

    def test_sample_count_is_honoured(self, tmp_path: Path) -> None:
        out = tmp_path / "ref.json"
        _run(BOOTSTRAP, "--samples", "25", "--out", str(out), cwd=PROJECT_ROOT)
        data = json.loads(out.read_text())
        assert all(len(values) == 25 for values in data.values())


class TestBuildReferenceFunction:
    def test_covers_every_numeric_feature(self) -> None:
        from scripts.bootstrap_reference import NUMERIC_COLUMNS, build_reference

        reference = build_reference(n_samples=40)
        for column in NUMERIC_COLUMNS:
            assert column in reference

    def test_includes_the_prediction_baseline(self) -> None:
        from scripts.bootstrap_reference import build_reference

        assert "prediction" in build_reference(n_samples=30)

    def test_is_reproducible_for_a_seed(self) -> None:
        from scripts.bootstrap_reference import build_reference

        assert build_reference(30, seed=5) == build_reference(30, seed=5)


class TestDiagramScript:
    @pytest.mark.skipif(
        __import__("importlib.util", fromlist=["util"]).find_spec("matplotlib") is None,
        reason="matplotlib is a dev-only dependency (requirements-dev.txt)",
    )
    def test_generates_a_png(self, tmp_path: Path) -> None:
        result = _run(DIAGRAM, cwd=tmp_path)
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "screenshots" / "architecture.png").exists()
