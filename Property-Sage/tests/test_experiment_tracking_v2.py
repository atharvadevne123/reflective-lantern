"""Tests for log_model_version and list_experiments."""

import pytest

from app.experiment_tracking import list_experiments, list_runs, log_model_version, log_run


@pytest.fixture
def exp_dir(tmp_path, monkeypatch):
    import app.experiment_tracking as et
    monkeypatch.setattr(et, "EXPERIMENT_DIR", tmp_path)
    return tmp_path


def test_log_model_version_returns_id(exp_dir):
    run_id = log_model_version("price_model", "1.0.0", {"r2": 0.9})
    assert isinstance(run_id, str)
    assert len(run_id) > 0


def test_log_model_version_stored(exp_dir):
    log_model_version("rental_model", "2.0.0", {"r2": 0.8})
    runs = list_runs("model_registry_rental_model")
    assert len(runs) >= 1
    assert runs[0]["params"]["version"] == "2.0.0"


def test_list_experiments_empty(exp_dir):
    assert list_experiments() == []


def test_list_experiments_after_log(exp_dir):
    log_run("alpha", {}, {"r2": 0.9})
    log_run("beta", {}, {"r2": 0.8})
    exps = list_experiments()
    assert "alpha" in exps
    assert "beta" in exps


def test_list_experiments_sorted(exp_dir):
    log_run("z_exp", {}, {"r2": 0.5})
    log_run("a_exp", {}, {"r2": 0.5})
    exps = list_experiments()
    assert exps == sorted(exps)


@pytest.mark.parametrize("version", ["1.0.0", "2.1.3", "20241030"])
def test_log_model_version_parametrized(exp_dir, version):
    run_id = log_model_version("m", version, {"score": 0.5})
    assert run_id is not None
