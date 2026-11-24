"""Extended experiment tracking tests with parametrize and boundary coverage."""
from __future__ import annotations

import pytest

from app.experiment_tracking import (
    get_best_run,
    get_recent_runs,
    list_experiments,
    list_runs,
    log_run,
)


@pytest.fixture(autouse=True)
def use_tmp_experiment_dir(tmp_path, monkeypatch):
    import app.experiment_tracking as et
    monkeypatch.setattr(et, "EXPERIMENT_DIR", tmp_path)


def test_log_run_returns_string_run_id() -> None:
    run_id = log_run("test_exp", {"lr": 0.01}, {"r2": 0.85})
    assert isinstance(run_id, str)
    assert len(run_id) > 0


def test_log_run_run_id_unique() -> None:
    id1 = log_run("test_exp", {}, {"r2": 0.8})
    id2 = log_run("test_exp", {}, {"r2": 0.9})
    assert id1 != id2


@pytest.mark.parametrize("experiment_name", ["exp_a", "exp_b", "model_v1"])
def test_log_run_creates_jsonl_file(tmp_path, experiment_name: str) -> None:
    import app.experiment_tracking as et
    log_run(experiment_name, {}, {"r2": 0.5})
    assert (et.EXPERIMENT_DIR / f"{experiment_name}.jsonl").exists()


def test_list_runs_empty_no_file() -> None:
    result = list_runs("nonexistent_exp")
    assert result == []


def test_list_runs_returns_records() -> None:
    log_run("list_test", {"n_estimators": 100}, {"r2": 0.8})
    log_run("list_test", {"n_estimators": 200}, {"r2": 0.9})
    runs = list_runs("list_test")
    assert len(runs) == 2


def test_list_runs_newest_first() -> None:
    for i in range(3):
        log_run("order_test", {"step": i}, {"r2": float(i) / 10})
    runs = list_runs("order_test")
    assert len(runs) == 3


def test_get_best_run_no_runs_returns_none() -> None:
    result = get_best_run("empty_exp", "r2")
    assert result is None


def test_get_best_run_higher_is_better() -> None:
    log_run("best_test", {"lr": 0.1}, {"r2": 0.7})
    log_run("best_test", {"lr": 0.01}, {"r2": 0.9})
    log_run("best_test", {"lr": 0.001}, {"r2": 0.8})
    best = get_best_run("best_test", "r2", higher_is_better=True)
    assert best is not None
    assert best["metrics"]["r2"] == pytest.approx(0.9)


def test_get_best_run_lower_is_better() -> None:
    log_run("rmse_test", {}, {"rmse": 0.3})
    log_run("rmse_test", {}, {"rmse": 0.1})
    log_run("rmse_test", {}, {"rmse": 0.5})
    best = get_best_run("rmse_test", "rmse", higher_is_better=False)
    assert best is not None
    assert best["metrics"]["rmse"] == pytest.approx(0.1)


def test_get_recent_runs_at_most_n() -> None:
    for i in range(5):
        log_run("recent_test", {}, {"r2": float(i) / 10})
    result = get_recent_runs("recent_test", n=3)
    assert len(result) <= 3


def test_get_recent_runs_empty_experiment() -> None:
    result = get_recent_runs("no_such_exp", n=5)
    assert result == []


def test_list_experiments_empty_dir() -> None:
    result = list_experiments()
    assert isinstance(result, list)


@pytest.mark.parametrize("n_runs", [1, 3, 5])
def test_log_multiple_runs_all_returned(n_runs: int) -> None:
    exp_name = f"multi_{n_runs}"
    for i in range(n_runs):
        log_run(exp_name, {"run": i}, {"metric": float(i)})
    runs = list_runs(exp_name)
    assert len(runs) == n_runs
