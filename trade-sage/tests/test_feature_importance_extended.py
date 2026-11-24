"""Extended feature importance tests with parametrize and boundary coverage."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.feature_importance import (
    extract_feature_importances,
    importance_as_dict,
    top_k_features,
)


def _make_rf_pipeline(n_features: int = 5) -> Pipeline:
    X = np.random.default_rng(42).random((100, n_features))
    y = (X[:, 0] > 0.5).astype(int)
    names = [f"f{i}" for i in range(n_features)]
    pipe = Pipeline([("rf", RandomForestClassifier(n_estimators=5, random_state=42))])
    pipe.fit(X, y)
    return pipe, names


def _make_lr_pipeline(n_features: int = 5) -> Pipeline:
    X = np.random.default_rng(0).random((100, n_features))
    y = (X[:, 0] > 0.5).astype(int)
    names = [f"feat_{i}" for i in range(n_features)]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(random_state=0)),
    ])
    pipe.fit(X, y)
    return pipe, names


def test_extract_feature_importances_returns_list() -> None:
    pipe, names = _make_rf_pipeline(5)
    result = extract_feature_importances(pipe, names)
    assert isinstance(result, list)


def test_extract_feature_importances_length_matches_features() -> None:
    pipe, names = _make_rf_pipeline(5)
    result = extract_feature_importances(pipe, names)
    assert len(result) == 5


def test_extract_feature_importances_sorted_descending() -> None:
    pipe, names = _make_rf_pipeline(5)
    result = extract_feature_importances(pipe, names)
    scores = [r[1] for r in result]
    assert scores == sorted(scores, reverse=True)


def test_extract_feature_importances_all_names_present() -> None:
    pipe, names = _make_rf_pipeline(5)
    result = extract_feature_importances(pipe, names)
    result_names = {r[0] for r in result}
    assert result_names == set(names)


def test_extract_feature_importances_scores_sum_to_one() -> None:
    pipe, names = _make_rf_pipeline(5)
    result = extract_feature_importances(pipe, names)
    total = sum(r[1] for r in result)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_extract_feature_importances_mismatched_names_returns_empty() -> None:
    pipe, names = _make_rf_pipeline(5)
    result = extract_feature_importances(pipe, names[:3])
    assert result == []


@pytest.mark.parametrize("n_features", [2, 5, 10, 20])
def test_extract_feature_importances_various_sizes(n_features: int) -> None:
    pipe, names = _make_rf_pipeline(n_features)
    result = extract_feature_importances(pipe, names)
    assert len(result) == n_features


def test_extract_feature_importances_logistic_regression() -> None:
    pipe, names = _make_lr_pipeline(5)
    result = extract_feature_importances(pipe, names)
    assert len(result) == 5


@pytest.mark.parametrize("k", [1, 3, 5])
def test_top_k_features_returns_at_most_k(k: int) -> None:
    pipe, names = _make_rf_pipeline(5)
    result = top_k_features(pipe, names, k=k)
    assert len(result) <= k


def test_top_k_features_larger_than_available() -> None:
    pipe, names = _make_rf_pipeline(3)
    result = top_k_features(pipe, names, k=10)
    assert len(result) == 3


def test_top_k_features_k_one_returns_highest() -> None:
    pipe, names = _make_rf_pipeline(5)
    all_pairs = extract_feature_importances(pipe, names)
    top = top_k_features(pipe, names, k=1)
    assert top[0] == all_pairs[0]


def test_importance_as_dict_returns_dict() -> None:
    pipe, names = _make_rf_pipeline(5)
    result = importance_as_dict(pipe, names)
    assert isinstance(result, dict)


def test_importance_as_dict_keys_match_names() -> None:
    pipe, names = _make_rf_pipeline(5)
    result = importance_as_dict(pipe, names)
    assert set(result.keys()) == set(names)


def test_importance_as_dict_values_non_negative() -> None:
    pipe, names = _make_rf_pipeline(5)
    result = importance_as_dict(pipe, names)
    assert all(v >= 0.0 for v in result.values())
