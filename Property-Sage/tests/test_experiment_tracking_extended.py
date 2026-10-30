"""Extended tests for experiment_tracking helpers."""

import json
import pytest

from app.experiment_tracking import get_recent_runs, list_runs, log_run


@pytest.fixture
def experiment_dir(tmp_path, monkeypatch):
    import app.experiment_tracking as et
    monkeypatch.setattr(et, "EXPERIMENT_DIR", tmp_path)
    return tmp_path


def test_get_recent_runs_empty(experiment_dir):
    assert get_recent_runs("test_exp") == []


def test_get_recent_runs_returns_n(experiment_dir):
    for i in range(5):
        log_run("exp", {"i": i}, {"metric": float(i)})
    runs = get_recent_runs("exp", n=3)
    assert len(runs) == 3


def test_get_recent_runs_newest_first(experiment_dir):
    log_run("exp2", {"order": 1}, {"val": 1.0})
    log_run("exp2", {"order": 2}, {"val": 2.0})
    runs = get_recent_runs("exp2", n=2)
    assert runs[0]["params"]["order"] == 2


@pytest.mark.parametrize("n", [1, 3, 5])
def test_get_recent_runs_parametrized(experiment_dir, n):
    for i in range(5):
        log_run("paramexp", {"i": i}, {"score": float(i)})
    runs = get_recent_runs("paramexp", n=n)
    assert len(runs) == n
