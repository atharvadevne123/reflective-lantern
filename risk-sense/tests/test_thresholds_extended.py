"""Extended threshold optimisation tests with parametrize and edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from app.thresholds import (
    apply_threshold,
    find_optimal_threshold,
    threshold_summary,
)


def _perfect_scores(n: int = 100) -> tuple[list[int], list[float]]:
    y_true = [0] * (n // 2) + [1] * (n // 2)
    y_scores = [0.1] * (n // 2) + [0.9] * (n // 2)
    return y_true, y_scores


@pytest.mark.parametrize("metric", ["f1", "precision", "recall", "accuracy"])
def test_find_optimal_threshold_valid_metrics(metric: str) -> None:
    y_true, y_scores = _perfect_scores()
    t = find_optimal_threshold(y_true, y_scores, metric=metric)
    assert 0.0 < t <= 1.0


def test_find_optimal_threshold_unknown_metric_raises() -> None:
    with pytest.raises(ValueError, match="Unknown metric"):
        y_true, y_scores = _perfect_scores()
        find_optimal_threshold(y_true, y_scores, metric="auc")


def test_find_optimal_threshold_on_perfect_separation() -> None:
    y_true, y_scores = _perfect_scores()
    t = find_optimal_threshold(y_true, y_scores, metric="f1")
    assert t <= 0.9


def test_find_optimal_threshold_all_same_scores() -> None:
    y_true = [0, 0, 1, 1]
    y_scores = [0.5, 0.5, 0.5, 0.5]
    t = find_optimal_threshold(y_true, y_scores, metric="f1")
    assert isinstance(t, float)
    assert 0.0 < t <= 1.0


def test_find_optimal_threshold_all_positives() -> None:
    y_true = [1, 1, 1, 1]
    y_scores = [0.8, 0.9, 0.7, 0.6]
    t = find_optimal_threshold(y_true, y_scores, metric="recall")
    assert isinstance(t, float)


def test_find_optimal_threshold_all_negatives() -> None:
    y_true = [0, 0, 0, 0]
    y_scores = [0.2, 0.3, 0.1, 0.4]
    t = find_optimal_threshold(y_true, y_scores, metric="precision")
    assert isinstance(t, float)


@pytest.mark.parametrize(
    "threshold,expected_pred",
    [
        (0.5, [0, 0, 1, 1]),
        (0.3, [0, 1, 1, 1]),
        (0.7, [0, 0, 1, 0]),
        (0.0, [1, 1, 1, 1]),
        (1.0, [0, 0, 0, 0]),
    ],
)
def test_apply_threshold(threshold: float, expected_pred: list[int]) -> None:
    scores = np.array([0.2, 0.4, 0.8, 0.6])
    preds = apply_threshold(scores, threshold)
    assert list(preds) == expected_pred


def test_apply_threshold_returns_binary() -> None:
    scores = np.array([0.1, 0.5, 0.9])
    preds = apply_threshold(scores, 0.5)
    assert set(preds).issubset({0, 1})


def test_threshold_summary_perfect_model() -> None:
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.8, 0.9]
    summary = threshold_summary(y_true, y_scores, threshold=0.5)
    assert summary["precision"] == pytest.approx(1.0, abs=1e-4)
    assert summary["recall"] == pytest.approx(1.0, abs=1e-4)
    assert summary["f1"] == pytest.approx(1.0, abs=1e-4)
    assert summary["accuracy"] == pytest.approx(1.0, abs=1e-4)


def test_threshold_summary_no_positives_predicted() -> None:
    y_true = [0, 1, 0, 1]
    y_scores = [0.1, 0.2, 0.3, 0.4]
    summary = threshold_summary(y_true, y_scores, threshold=0.9)
    assert summary["precision"] == pytest.approx(0.0, abs=1e-4)
    assert summary["recall"] == pytest.approx(0.0, abs=1e-4)


def test_threshold_summary_has_all_keys() -> None:
    y_true, y_scores = _perfect_scores()
    summary = threshold_summary(y_true, y_scores, threshold=0.5)
    for key in ("threshold", "precision", "recall", "f1", "accuracy"):
        assert key in summary


@pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_threshold_summary_values_in_range(threshold: float) -> None:
    y_true, y_scores = _perfect_scores()
    summary = threshold_summary(y_true, y_scores, threshold=threshold)
    for key in ("precision", "recall", "f1", "accuracy"):
        assert 0.0 <= summary[key] <= 1.0


def test_threshold_summary_threshold_field_matches_input() -> None:
    y_true, y_scores = _perfect_scores()
    summary = threshold_summary(y_true, y_scores, threshold=0.75)
    assert summary["threshold"] == pytest.approx(0.75, abs=1e-4)
