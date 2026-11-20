"""Tests for the decision threshold optimisation module."""

from __future__ import annotations

import numpy as np
import pytest

from app.thresholds import (
    apply_threshold,
    find_optimal_threshold,
    threshold_summary,
)


class TestFindOptimalThreshold:
    def test_returns_float_in_range(self) -> None:
        y_true = [0, 1, 0, 1, 1, 0]
        y_scores = [0.2, 0.8, 0.3, 0.7, 0.9, 0.1]
        t = find_optimal_threshold(y_true, y_scores)
        assert 0.0 < t < 1.0

    def test_perfect_separator_finds_good_threshold(self) -> None:
        y_true = [0] * 50 + [1] * 50
        y_scores = [0.1] * 50 + [0.9] * 50
        t = find_optimal_threshold(y_true, y_scores, metric="f1")
        assert t < 0.5

    @pytest.mark.parametrize("metric", ["f1", "precision", "recall", "accuracy"])
    def test_valid_metrics(self, metric: str) -> None:
        y_true = [0, 1, 0, 1]
        y_scores = [0.3, 0.7, 0.4, 0.8]
        t = find_optimal_threshold(y_true, y_scores, metric=metric)
        assert isinstance(t, float)

    def test_unknown_metric_raises(self) -> None:
        y_true = [0, 1]
        y_scores = [0.3, 0.8]
        with pytest.raises(ValueError):
            find_optimal_threshold(y_true, y_scores, metric="kappa")


class TestApplyThreshold:
    def test_above_threshold_is_one(self) -> None:
        scores = np.array([0.6, 0.8])
        result = apply_threshold(scores, 0.5)
        assert list(result) == [1, 1]

    def test_below_threshold_is_zero(self) -> None:
        scores = np.array([0.3, 0.1])
        result = apply_threshold(scores, 0.5)
        assert list(result) == [0, 0]

    def test_equal_to_threshold_is_one(self) -> None:
        scores = np.array([0.5])
        result = apply_threshold(scores, 0.5)
        assert result[0] == 1


class TestThresholdSummary:
    def test_perfect_classifier(self) -> None:
        y_true = [0, 1, 0, 1]
        y_scores = [0.1, 0.9, 0.2, 0.8]
        summary = threshold_summary(y_true, y_scores, threshold=0.5)
        assert summary["precision"] == pytest.approx(1.0)
        assert summary["recall"] == pytest.approx(1.0)
        assert summary["f1"] == pytest.approx(1.0)
        assert summary["accuracy"] == pytest.approx(1.0)

    def test_summary_has_required_keys(self) -> None:
        y_true = [0, 1]
        y_scores = [0.3, 0.7]
        summary = threshold_summary(y_true, y_scores, threshold=0.5)
        for key in ("threshold", "precision", "recall", "f1", "accuracy"):
            assert key in summary

    def test_threshold_stored_in_summary(self) -> None:
        y_true = [0, 1]
        y_scores = [0.3, 0.7]
        summary = threshold_summary(y_true, y_scores, threshold=0.3)
        assert summary["threshold"] == pytest.approx(0.3)
