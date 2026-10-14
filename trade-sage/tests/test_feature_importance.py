"""Tests for feature importance extraction utilities."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from app.feature_importance import extract_feature_importances, top_k_features

FEATURE_NAMES = [f"f{i}" for i in range(10)]


@pytest.fixture(scope="module")
def fitted_rf_pipeline():
    rng = np.random.default_rng(0)
    X = rng.random((200, 10))
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    clf = RandomForestClassifier(n_estimators=20, random_state=0)
    pipe = Pipeline([("clf", clf)])
    pipe.fit(X, y)
    return pipe


def test_returns_correct_number_of_features(fitted_rf_pipeline):
    pairs = extract_feature_importances(fitted_rf_pipeline, FEATURE_NAMES)
    assert len(pairs) == len(FEATURE_NAMES)


def test_importances_sum_to_one(fitted_rf_pipeline):
    pairs = extract_feature_importances(fitted_rf_pipeline, FEATURE_NAMES)
    total = sum(v for _, v in pairs)
    assert abs(total - 1.0) < 1e-5


def test_sorted_descending(fitted_rf_pipeline):
    pairs = extract_feature_importances(fitted_rf_pipeline, FEATURE_NAMES)
    values = [v for _, v in pairs]
    assert values == sorted(values, reverse=True)


def test_feature_names_preserved(fitted_rf_pipeline):
    pairs = extract_feature_importances(fitted_rf_pipeline, FEATURE_NAMES)
    returned_names = {name for name, _ in pairs}
    assert returned_names == set(FEATURE_NAMES)


def test_mismatched_names_returns_empty(fitted_rf_pipeline):
    result = extract_feature_importances(fitted_rf_pipeline, ["a", "b"])
    assert result == []


def test_top_k_returns_at_most_k(fitted_rf_pipeline):
    result = top_k_features(fitted_rf_pipeline, FEATURE_NAMES, k=3)
    assert len(result) <= 3


def test_top_k_are_highest(fitted_rf_pipeline):
    all_pairs = extract_feature_importances(fitted_rf_pipeline, FEATURE_NAMES)
    top3 = top_k_features(fitted_rf_pipeline, FEATURE_NAMES, k=3)
    top3_names = {n for n, _ in top3}
    expected = {n for n, _ in all_pairs[:3]}
    assert top3_names == expected
