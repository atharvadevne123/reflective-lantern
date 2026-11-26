"""Extended decision threshold tests — second pass with parametrize."""

from __future__ import annotations

import numpy as np
import pytest

from app.thresholds import apply_threshold, find_optimal_threshold, threshold_summary


def _binary_scenario(n_pos: int = 50, n_neg: int = 50, seed: int = 0) -> tuple[list[int], list[float]]:
    rng = np.random.default_rng(seed)
    y_true = [1] * n_pos + [0] * n_neg
    scores_pos = rng.uniform(0.6, 1.0, n_pos).tolist()
    scores_neg = rng.uniform(0.0, 0.5, n_neg).tolist()
    return y_true, scores_pos + scores_neg


def test_find_optimal_threshold_returns_float() -> None:
    y_true, scores = _binary_scenario()
    t = find_optimal_threshold(y_true, scores)
    assert isinstance(t, float)


def test_find_optimal_threshold_in_valid_range() -> None:
    y_true, scores = _binary_scenario()
    t = find_optimal_threshold(y_true, scores)
    assert 0.0 < t < 1.0


@pytest.mark.parametrize("metric", ["f1", "precision", "recall", "accuracy"])
def test_find_optimal_threshold_various_metrics(metric: str) -> None:
    y_true, scores = _binary_scenario()
    t = find_optimal_threshold(y_true, scores, metric=metric)
    assert 0.0 < t < 1.0


def test_find_optimal_threshold_all_positives() -> None:
    y_true = [1] * 100
    scores = [0.9] * 100
    t = find_optimal_threshold(y_true, scores)
    assert isinstance(t, float)


def test_apply_threshold_binary_output() -> None:
    scores = np.array([0.1, 0.5, 0.8, 0.3])
    preds = apply_threshold(scores, 0.5)
    assert set(preds).issubset({0, 1})


def test_apply_threshold_above_all_gives_all_ones() -> None:
    scores = np.array([0.9, 0.95, 0.85])
    preds = apply_threshold(scores, 0.5)
    assert all(preds == 1)


def test_apply_threshold_above_all_gives_all_zeros() -> None:
    scores = np.array([0.1, 0.2, 0.3])
    preds = apply_threshold(scores, 0.9)
    assert all(preds == 0)


@pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_apply_threshold_various_thresholds(threshold: float) -> None:
    scores = np.linspace(0.0, 1.0, 100)
    preds = apply_threshold(scores, threshold)
    assert len(preds) == len(scores)
    assert set(preds).issubset({0, 1})


def test_threshold_summary_has_required_keys() -> None:
    y_true, scores = _binary_scenario()
    summary = threshold_summary(y_true, scores, 0.5)
    for key in ("threshold", "precision", "recall", "f1", "accuracy"):
        assert key in summary


def test_threshold_summary_values_in_unit_interval() -> None:
    y_true, scores = _binary_scenario()
    summary = threshold_summary(y_true, scores, 0.5)
    for key in ("precision", "recall", "f1", "accuracy"):
        assert 0.0 <= summary[key] <= 1.0


def test_threshold_summary_at_zero_threshold_all_positives() -> None:
    y_true = [1] * 10 + [0] * 10
    scores = [0.9] * 10 + [0.1] * 10
    summary = threshold_summary(y_true, scores, 0.5)
    assert summary["accuracy"] > 0.5


def test_threshold_summary_threshold_stored_correctly() -> None:
    y_true, scores = _binary_scenario()
    summary = threshold_summary(y_true, scores, 0.7)
    assert summary["threshold"] == pytest.approx(0.7, abs=0.001)
