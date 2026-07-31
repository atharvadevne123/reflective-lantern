"""Tests for app/pipeline_utils.py."""

from __future__ import annotations

import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.pipeline_utils import (
    bundle_pipeline_info,
    clone_params,
    describe_pipeline,
    get_step,
    get_step_names,
    has_step,
    pipeline_param_count,
)


def make_test_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ])


def test_get_step_names() -> None:
    pipe = make_test_pipeline()
    names = get_step_names(pipe)
    assert names == ["scaler", "regressor"]


def test_get_step_names_non_pipeline() -> None:
    assert get_step_names("not a pipeline") == []


def test_has_step_true() -> None:
    pipe = make_test_pipeline()
    assert has_step(pipe, "scaler") is True


def test_has_step_false() -> None:
    pipe = make_test_pipeline()
    assert has_step(pipe, "nonexistent") is False


def test_get_step_returns_estimator() -> None:
    pipe = make_test_pipeline()
    step = get_step(pipe, "scaler")
    assert isinstance(step, StandardScaler)


def test_get_step_missing() -> None:
    pipe = make_test_pipeline()
    assert get_step(pipe, "missing") is None


def test_get_step_non_pipeline() -> None:
    assert get_step("not a pipeline", "step") is None


def test_pipeline_param_count_positive() -> None:
    pipe = make_test_pipeline()
    count = pipeline_param_count(pipe)
    assert count > 0


def test_pipeline_param_count_non_pipeline() -> None:
    assert pipeline_param_count(object()) == 0


def test_describe_pipeline_structure() -> None:
    pipe = make_test_pipeline()
    desc = describe_pipeline(pipe)
    assert desc["n_steps"] == 2
    assert len(desc["steps"]) == 2
    assert desc["steps"][0]["name"] == "scaler"
    assert desc["steps"][1]["name"] == "regressor"


def test_describe_pipeline_class_names() -> None:
    pipe = make_test_pipeline()
    desc = describe_pipeline(pipe)
    assert desc["steps"][0]["class"] == "StandardScaler"
    assert desc["steps"][1]["class"] == "LinearRegression"


def test_describe_pipeline_non_pipeline() -> None:
    desc = describe_pipeline(object())
    assert desc["n_steps"] == 0


def test_clone_params_dict() -> None:
    pipe = make_test_pipeline()
    params = clone_params(pipe)
    assert isinstance(params, dict)
    assert len(params) > 0


def test_clone_params_non_pipeline() -> None:
    result = clone_params(object())
    assert result == {}


def test_bundle_pipeline_info_with_model() -> None:
    pipe = make_test_pipeline()
    info = bundle_pipeline_info({"model": pipe})
    assert "n_steps" in info
    assert info["n_steps"] == 2


def test_bundle_pipeline_info_empty_bundle() -> None:
    info = bundle_pipeline_info({})
    assert "error" in info


@pytest.mark.parametrize("step_name", ["scaler", "regressor"])
def test_has_step_parametrize(step_name) -> None:
    pipe = make_test_pipeline()
    assert has_step(pipe, step_name) is True


@pytest.mark.parametrize("step_name", ["scaler", "regressor"])
def test_has_step_parametrized(step_name: str) -> None:
    pipe = make_test_pipeline()
    assert has_step(pipe, step_name) is True


@pytest.mark.parametrize("invalid_step", ["normalizer", "pca", "forest"])
def test_has_step_false_parametrized(invalid_step: str) -> None:
    pipe = make_test_pipeline()
    assert has_step(pipe, invalid_step) is False


def test_get_step_none_for_missing() -> None:
    pipe = make_test_pipeline()
    assert get_step(pipe, "nonexistent") is None


def test_pipeline_param_count_is_positive() -> None:
    pipe = make_test_pipeline()
    count = pipeline_param_count(pipe)
    assert count > 0


def test_pipeline_param_count_no_pipeline() -> None:
    assert pipeline_param_count("not a pipeline") == 0


def test_describe_pipeline_has_steps_key() -> None:
    pipe = make_test_pipeline()
    desc = describe_pipeline(pipe)
    assert "steps" in desc
    assert desc["n_steps"] == 2


def test_describe_pipeline_non_pipeline_returns_empty() -> None:
    desc = describe_pipeline("not a pipeline")
    assert desc.get("n_steps", 0) == 0 or "steps" not in desc or desc["steps"] == []


def test_bundle_pipeline_info_has_step_count() -> None:
    pipe = make_test_pipeline()
    info = bundle_pipeline_info({"model": pipe})
    assert "n_steps" in info
    assert info["n_steps"] == 2


def test_bundle_pipeline_info_has_steps() -> None:
    pipe = make_test_pipeline()
    info = bundle_pipeline_info({"model": pipe})
    assert "steps" in info
    step_names = [s["name"] for s in info["steps"]]
    assert "scaler" in step_names


def test_bundle_pipeline_info_no_model_returns_error() -> None:
    info = bundle_pipeline_info({})
    assert "error" in info


def test_clone_params_returns_dict() -> None:
    pipe = make_test_pipeline()
    params = clone_params(pipe)
    assert isinstance(params, dict)


@pytest.mark.parametrize("step_name", ["scaler", "regressor"])
def test_has_step_parametrized(step_name: str) -> None:
    pipe = make_test_pipeline()
    assert has_step(pipe, step_name) is True


def test_get_step_names_length() -> None:
    pipe = make_test_pipeline()
    assert len(get_step_names(pipe)) == 2


def test_pipeline_param_count_is_int() -> None:
    pipe = make_test_pipeline()
    count = pipeline_param_count(pipe)
    assert isinstance(count, int)
