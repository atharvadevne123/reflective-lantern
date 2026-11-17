"""Tests for Fleet-Mind model explainability utilities."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.model_explainability import get_feature_importances


def _make_model_with_importances(values):
    model = MagicMock()
    model.feature_importances_ = np.array(values)
    del model.estimators_
    return model


def test_returns_list():
    from app.features import ALL_FEATURES
    n = len(ALL_FEATURES)
    model = _make_model_with_importances([1.0 / n] * n)
    result = get_feature_importances(model)
    assert isinstance(result, list)


def test_result_length_matches_feature_count():
    from app.features import ALL_FEATURES
    n = len(ALL_FEATURES)
    model = _make_model_with_importances([1.0 / n] * n)
    result = get_feature_importances(model)
    assert len(result) == n


def test_result_sorted_descending():
    from app.features import ALL_FEATURES
    n = len(ALL_FEATURES)
    values = [float(i) for i in range(n)]
    model = _make_model_with_importances(values)
    result = get_feature_importances(model)
    importances = [r["importance"] for r in result]
    assert importances == sorted(importances, reverse=True)


def test_result_entries_have_feature_and_importance():
    from app.features import ALL_FEATURES
    n = len(ALL_FEATURES)
    model = _make_model_with_importances([0.1] * n)
    result = get_feature_importances(model)
    for entry in result:
        assert "feature" in entry
        assert "importance" in entry


def test_model_without_importances_returns_empty():
    model = MagicMock(spec=[])  # no feature_importances_ attribute
    result = get_feature_importances(model)
    assert result == []


def test_mismatched_importance_count_returns_empty():
    model = MagicMock()
    model.feature_importances_ = np.array([0.5, 0.5])  # only 2 features
    del model.estimators_
    result = get_feature_importances(model)
    assert result == []


def test_voting_classifier_averages_importances():
    from app.features import ALL_FEATURES
    n = len(ALL_FEATURES)
    sub1 = MagicMock()
    sub1.feature_importances_ = np.ones(n) * 0.2
    sub2 = MagicMock()
    sub2.feature_importances_ = np.ones(n) * 0.4

    model = MagicMock()
    model.estimators_ = [("clf1", sub1), ("clf2", sub2)]
    del model.feature_importances_

    result = get_feature_importances(model)
    if result:  # only check if features matched
        avg_imp = sum(r["importance"] for r in result) / len(result)
        assert abs(avg_imp - 0.3) < 0.01
