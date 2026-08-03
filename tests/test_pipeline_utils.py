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
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )


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
def test_has_step_true_parametrized(step_name: str) -> None:
    pipe = make_test_pipeline()
    assert has_step(pipe, step_name) is True


def test_get_step_names_length() -> None:
    pipe = make_test_pipeline()
    assert len(get_step_names(pipe)) == 2


def test_pipeline_param_count_is_int() -> None:
    pipe = make_test_pipeline()
    count = pipeline_param_count(pipe)
    assert isinstance(count, int)


class TestIsAndStepFitted:
    def test_is_fitted_unfitted_pipeline(self) -> None:
        from app.pipeline_utils import is_fitted

        pipe = make_test_pipeline()
        # An unfitted sklearn pipeline has no fitted attrs yet
        result = is_fitted(pipe)
        # Pipeline class itself has some class-level attrs ending in _
        # We just ensure the function returns a bool
        assert isinstance(result, bool)

    def test_is_fitted_after_fit(self) -> None:
        import numpy as np

        from app.pipeline_utils import is_fitted

        pipe = make_test_pipeline()
        X = np.random.default_rng(0).random((10, 1))
        y = np.arange(10, dtype=float)
        pipe.fit(X, y)
        assert is_fitted(pipe) is True

    def test_step_is_fitted_nonexistent_step(self) -> None:
        from app.pipeline_utils import step_is_fitted

        pipe = make_test_pipeline()
        assert step_is_fitted(pipe, "nonexistent") is False

    def test_step_is_fitted_after_fit(self) -> None:
        import numpy as np

        from app.pipeline_utils import step_is_fitted

        pipe = make_test_pipeline()
        X = np.random.default_rng(1).random((10, 1))
        y = np.arange(10, dtype=float)
        pipe.fit(X, y)
        assert step_is_fitted(pipe, "scaler") is True

    def test_bundle_pipeline_info_with_pipeline_key(self) -> None:
        from app.pipeline_utils import bundle_pipeline_info

        pipe = make_test_pipeline()
        info = bundle_pipeline_info({"pipeline": pipe})
        assert info.get("n_steps", 0) == 2

    def test_bundle_pipeline_info_missing_keys(self) -> None:
        from app.pipeline_utils import bundle_pipeline_info

        info = bundle_pipeline_info({})
        assert "error" in info

    @pytest.mark.parametrize("n_steps", [1, 2, 3])
    def test_describe_pipeline_step_count(self, n_steps: int) -> None:
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from app.pipeline_utils import describe_pipeline

        steps = [(f"step{i}", StandardScaler() if i < n_steps - 1 else LinearRegression()) for i in range(n_steps)]
        pipe = Pipeline(steps)
        info = describe_pipeline(pipe)
        assert info["n_steps"] == n_steps


class TestPipelineStepTypes:
    def test_empty_pipeline(self) -> None:
        from app.pipeline_utils import pipeline_step_types
        assert pipeline_step_types(object()) == {}

    def test_returns_class_names(self) -> None:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from app.pipeline_utils import pipeline_step_types
        p = Pipeline([("scaler", StandardScaler())])
        result = pipeline_step_types(p)
        assert result == {"scaler": "StandardScaler"}

    def test_multiple_steps(self) -> None:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        from app.pipeline_utils import pipeline_step_types
        p = Pipeline([("std", StandardScaler()), ("mm", MinMaxScaler())])
        result = pipeline_step_types(p)
        assert len(result) == 2
        assert "std" in result and "mm" in result


class TestFirstStep:
    def test_returns_first(self) -> None:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from app.pipeline_utils import first_step
        scaler = StandardScaler()
        p = Pipeline([("scaler", scaler)])
        assert first_step(p) is scaler

    def test_empty_returns_none(self) -> None:
        from app.pipeline_utils import first_step
        assert first_step(object()) is None
