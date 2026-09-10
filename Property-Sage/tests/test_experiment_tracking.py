"""Tests for the experiment tracking module."""


import pytest

from app.experiment_tracking import get_best_run, list_runs, log_run


@pytest.fixture(autouse=True)
def tmp_experiment_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_DIR", str(tmp_path / "experiments"))
    import app.experiment_tracking as et
    et.EXPERIMENT_DIR = tmp_path / "experiments"
    yield


def test_log_run_returns_run_id():
    run_id = log_run("test_exp", params={"lr": 0.01}, metrics={"r2": 0.85})
    assert isinstance(run_id, str)
    assert len(run_id) == 8


def test_log_run_persists_to_file():
    import app.experiment_tracking as et
    log_run("test_exp2", params={}, metrics={"rmse": 5000.0})
    log_path = et.EXPERIMENT_DIR / "test_exp2.jsonl"
    assert log_path.exists()


def test_list_runs_returns_all():
    for i in range(3):
        log_run("multi_exp", params={"i": i}, metrics={"r2": 0.7 + i * 0.05})
    runs = list_runs("multi_exp")
    assert len(runs) == 3


def test_list_runs_empty_for_unknown():
    runs = list_runs("nonexistent_experiment")
    assert runs == []


def test_get_best_run_highest():
    log_run("price_exp", params={}, metrics={"price_r2_mean": 0.80})
    log_run("price_exp", params={}, metrics={"price_r2_mean": 0.92})
    log_run("price_exp", params={}, metrics={"price_r2_mean": 0.75})
    best = get_best_run("price_exp", metric="price_r2_mean", higher_is_better=True)
    assert best is not None
    assert best["metrics"]["price_r2_mean"] == pytest.approx(0.92)


def test_get_best_run_lowest():
    log_run("rmse_exp", params={}, metrics={"rmse": 10000.0})
    log_run("rmse_exp", params={}, metrics={"rmse": 5000.0})
    best = get_best_run("rmse_exp", metric="rmse", higher_is_better=False)
    assert best["metrics"]["rmse"] == pytest.approx(5000.0)


def test_get_best_run_none_for_missing_experiment():
    result = get_best_run("missing_exp", metric="r2")
    assert result is None


def test_run_record_has_required_keys():
    log_run("key_exp", params={"a": 1}, metrics={"r2": 0.9}, tags={"env": "test"})
    runs = list_runs("key_exp")
    r = runs[0]
    for key in ("run_id", "experiment_name", "timestamp", "params", "metrics", "tags"):
        assert key in r
