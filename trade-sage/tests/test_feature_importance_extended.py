"""Extended tests for feature importance utilities including new functions."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.feature_importance import (
    cumulative_importance,
    importance_as_dict,
)

FEATURE_NAMES = [f"f{i}" for i in range(10)]


@pytest.fixture(scope="module")
def fitted_pipeline() -> Pipeline:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 10))
    y = (X[:, 0] + rng.standard_normal(200) > 0).astype(int)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=20, random_state=0))])
    pipe.fit(X, y)
    return pipe


class TestImportanceAsDict:
    def test_returns_dict(self, fitted_pipeline: Pipeline) -> None:
        result = importance_as_dict(fitted_pipeline, FEATURE_NAMES)
        assert isinstance(result, dict)

    def test_all_feature_names_present(self, fitted_pipeline: Pipeline) -> None:
        result = importance_as_dict(fitted_pipeline, FEATURE_NAMES)
        for name in FEATURE_NAMES:
            assert name in result

    def test_values_are_floats(self, fitted_pipeline: Pipeline) -> None:
        result = importance_as_dict(fitted_pipeline, FEATURE_NAMES)
        for v in result.values():
            assert isinstance(v, float)

    def test_values_sum_to_approximately_one(self, fitted_pipeline: Pipeline) -> None:
        result = importance_as_dict(fitted_pipeline, FEATURE_NAMES)
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-3

    def test_empty_feature_names_returns_empty(self, fitted_pipeline: Pipeline) -> None:
        result = importance_as_dict(fitted_pipeline, [])
        assert result == {}


class TestCumulativeImportance:
    def test_returns_list_of_strings(self, fitted_pipeline: Pipeline) -> None:
        result = cumulative_importance(fitted_pipeline, FEATURE_NAMES)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)

    def test_threshold_90_covers_90_percent(self, fitted_pipeline: Pipeline) -> None:
        selected = cumulative_importance(fitted_pipeline, FEATURE_NAMES, threshold=0.90)
        assert len(selected) >= 1
        assert len(selected) <= len(FEATURE_NAMES)

    def test_threshold_100_returns_all_features(self, fitted_pipeline: Pipeline) -> None:
        selected = cumulative_importance(fitted_pipeline, FEATURE_NAMES, threshold=1.0)
        assert len(selected) == len(FEATURE_NAMES)

    def test_threshold_0_returns_one_feature(self, fitted_pipeline: Pipeline) -> None:
        selected = cumulative_importance(fitted_pipeline, FEATURE_NAMES, threshold=0.0)
        assert len(selected) == 1

    def test_selected_features_are_subset_of_names(self, fitted_pipeline: Pipeline) -> None:
        selected = cumulative_importance(fitted_pipeline, FEATURE_NAMES)
        for name in selected:
            assert name in FEATURE_NAMES

    def test_empty_pipeline_returns_empty(self) -> None:
        result = cumulative_importance(None, [])
        assert result == []

    @pytest.mark.parametrize("threshold", [0.5, 0.75, 0.9, 0.95])
    def test_various_thresholds(self, fitted_pipeline: Pipeline, threshold: float) -> None:
        selected = cumulative_importance(fitted_pipeline, FEATURE_NAMES, threshold=threshold)
        assert len(selected) >= 1
