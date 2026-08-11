"""Tests for app/pipeline_utils.py."""

from __future__ import annotations

import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.pipeline_utils import (
    bundle_pipeline_info,
    clone_params,
    count_fitted_steps,
    describe_pipeline,
    extract_step_classes,
    get_step,
    get_step_names,
    has_step,
    pipeline_has_preprocessor,
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


class _FakePipeline:
    def __init__(self, steps):
        self.steps = steps


class TestCountPipelineSteps:
    def test_two_steps(self) -> None:
        from app.pipeline_utils import count_pipeline_steps

        pipe = _FakePipeline([("a", object()), ("b", object())])
        assert count_pipeline_steps(pipe) == 2

    def test_no_steps_attr(self) -> None:
        from app.pipeline_utils import count_pipeline_steps

        assert count_pipeline_steps(object()) == 0

    def test_empty_steps(self) -> None:
        from app.pipeline_utils import count_pipeline_steps

        assert count_pipeline_steps(_FakePipeline([])) == 0


class TestLastStep:
    def test_returns_last(self) -> None:
        from app.pipeline_utils import last_step

        obj = object()
        pipe = _FakePipeline([("a", object()), ("b", obj)])
        assert last_step(pipe) is obj

    def test_empty_returns_none(self) -> None:
        from app.pipeline_utils import last_step

        assert last_step(_FakePipeline([])) is None

    def test_no_steps_attr(self) -> None:
        from app.pipeline_utils import last_step

        assert last_step(object()) is None


class TestStepNamesToSet:
    def test_basic(self) -> None:
        from app.pipeline_utils import step_names_to_set

        pipe = _FakePipeline([("scaler", object()), ("model", object())])
        result = step_names_to_set(pipe)
        assert result == {"scaler", "model"}

    def test_empty(self) -> None:
        from app.pipeline_utils import step_names_to_set

        assert step_names_to_set(_FakePipeline([])) == set()

    def test_no_steps_attr(self) -> None:
        from app.pipeline_utils import step_names_to_set

        assert step_names_to_set(object()) == set()


class TestPipelineMemoryUsageKb:
    def test_positive_result(self) -> None:
        from app.pipeline_utils import pipeline_memory_usage_kb

        result = pipeline_memory_usage_kb({"a": [1, 2, 3]})
        assert result > 0

    def test_larger_object_bigger(self) -> None:
        from app.pipeline_utils import pipeline_memory_usage_kb

        small = pipeline_memory_usage_kb([1])
        large = pipeline_memory_usage_kb(list(range(1000)))
        assert large > small


def test_count_fitted_steps_unfitted() -> None:
    pipe = make_test_pipeline()
    assert count_fitted_steps(pipe) == 0


def test_count_fitted_steps_non_pipeline() -> None:
    assert count_fitted_steps("not a pipeline") == 0


def test_pipeline_has_preprocessor_true() -> None:
    pipe = make_test_pipeline()
    assert pipeline_has_preprocessor(pipe) is True


def test_pipeline_has_preprocessor_false() -> None:
    from sklearn.linear_model import Ridge

    pipe = Pipeline([("ridge", Ridge())])
    assert pipeline_has_preprocessor(pipe) is False


def test_pipeline_has_preprocessor_non_pipeline() -> None:
    assert pipeline_has_preprocessor("not a pipeline") is False


def test_extract_step_classes_returns_names() -> None:
    pipe = make_test_pipeline()
    classes = extract_step_classes(pipe)
    assert classes == ["StandardScaler", "LinearRegression"]


def test_extract_step_classes_non_pipeline() -> None:
    assert extract_step_classes("not a pipeline") == []


def test_extract_step_classes_length_matches_steps() -> None:
    pipe = make_test_pipeline()
    assert len(extract_step_classes(pipe)) == len(get_step_names(pipe))


@pytest.mark.parametrize(
    "keyword_step",
    [
        ("scaler", StandardScaler()),
        ("normalizer", StandardScaler()),
        ("encoder", StandardScaler()),
    ],
)
def test_pipeline_has_preprocessor_keyword_variants(keyword_step) -> None:
    name, estimator = keyword_step
    pipe = Pipeline([(name, estimator), ("reg", LinearRegression())])
    assert pipeline_has_preprocessor(pipe) is True


class TestPipelineStepIndex:
    def test_first_step_index(self) -> None:
        from app.pipeline_utils import pipeline_step_index

        pipe = make_test_pipeline()
        assert pipeline_step_index(pipe, "scaler") == 0

    def test_last_step_index(self) -> None:
        from app.pipeline_utils import pipeline_step_index

        pipe = make_test_pipeline()
        assert pipeline_step_index(pipe, "regressor") == 1

    def test_unknown_step_raises(self) -> None:
        from app.pipeline_utils import pipeline_step_index

        pipe = make_test_pipeline()
        with pytest.raises(KeyError):
            pipeline_step_index(pipe, "nonexistent")

    @pytest.mark.parametrize("step_name,expected", [("scaler", 0), ("regressor", 1)])
    def test_parametrized(self, step_name, expected) -> None:
        from app.pipeline_utils import pipeline_step_index

        assert pipeline_step_index(make_test_pipeline(), step_name) == expected


class TestPipelineStepBefore:
    def test_first_step_has_no_predecessors(self) -> None:
        from app.pipeline_utils import pipeline_step_before

        assert pipeline_step_before(make_test_pipeline(), "scaler") == []

    def test_last_step_predecessors(self) -> None:
        from app.pipeline_utils import pipeline_step_before

        assert pipeline_step_before(make_test_pipeline(), "regressor") == ["scaler"]

    def test_unknown_step_raises(self) -> None:
        from app.pipeline_utils import pipeline_step_before

        with pytest.raises(KeyError):
            pipeline_step_before(make_test_pipeline(), "missing")


class TestPipelineStepAfter:
    def test_last_step_has_no_successors(self) -> None:
        from app.pipeline_utils import pipeline_step_after

        assert pipeline_step_after(make_test_pipeline(), "regressor") == []

    def test_first_step_successors(self) -> None:
        from app.pipeline_utils import pipeline_step_after

        assert pipeline_step_after(make_test_pipeline(), "scaler") == ["regressor"]

    def test_unknown_step_raises(self) -> None:
        from app.pipeline_utils import pipeline_step_after

        with pytest.raises(KeyError):
            pipeline_step_after(make_test_pipeline(), "missing")

    @pytest.mark.parametrize(
        "step_name,expected",
        [("scaler", ["regressor"]), ("regressor", [])],
    )
    def test_parametrized(self, step_name, expected) -> None:
        from app.pipeline_utils import pipeline_step_after

        assert pipeline_step_after(make_test_pipeline(), step_name) == expected


# ---------------------------------------------------------------------------
# Tests for count_pipeline_params, pipeline_feature_count, is_pipeline_fitted
# ---------------------------------------------------------------------------


class TestCountPipelineParams:
    def test_unfitted_pipeline(self) -> None:
        from app.features import build_feature_pipeline
        from app.pipeline_utils import count_pipeline_params

        pipeline = build_feature_pipeline()
        result = count_pipeline_params(pipeline)
        assert isinstance(result, int)
        assert result > 0

    def test_non_pipeline_object(self) -> None:
        from app.pipeline_utils import count_pipeline_params

        assert count_pipeline_params(object()) == 0


class TestPipelineFeatureCount:
    def test_unfitted_returns_zero(self) -> None:
        from app.features import build_feature_pipeline
        from app.pipeline_utils import pipeline_feature_count

        pipeline = build_feature_pipeline()
        assert pipeline_feature_count(pipeline) == 0

    def test_non_pipeline_returns_zero(self) -> None:
        from app.pipeline_utils import pipeline_feature_count

        assert pipeline_feature_count(object()) == 0


class TestIsPipelineFitted:
    def test_unfitted_pipeline_returns_false(self) -> None:
        from app.features import build_feature_pipeline
        from app.pipeline_utils import is_pipeline_fitted

        pipeline = build_feature_pipeline()
        assert not is_pipeline_fitted(pipeline)

    def test_fitted_step_returns_true(self) -> None:
        import pandas as pd

        from app.features import DropNonNumeric
        from app.pipeline_utils import is_pipeline_fitted

        step = DropNonNumeric()
        step.fit(pd.DataFrame({"a": [1.0], "b": [2.0]}))
        assert is_pipeline_fitted(step)
