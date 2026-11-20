"""Decision threshold optimisation for credit risk models."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def find_optimal_threshold(
    y_true: list[int] | np.ndarray,
    y_scores: list[float] | np.ndarray,
    metric: str = "f1",
) -> float:
    """Find the classification threshold that maximises the given metric.

    Searches over a grid of candidate thresholds [0.05, 0.95] in 0.05 steps.
    Returns the threshold with the best score; ties broken by selecting the
    lower threshold (prefer recall over precision in a risk context).
    """
    y_true_arr = np.asarray(y_true)
    y_scores_arr = np.asarray(y_scores)

    candidates = np.arange(0.05, 0.96, 0.05)
    best_threshold = 0.5
    best_score = -1.0

    for t in candidates:
        y_pred = (y_scores_arr >= t).astype(int)
        score = _metric_score(y_true_arr, y_pred, metric)
        if score > best_score:
            best_score = score
            best_threshold = float(t)

    logger.debug("Best threshold %.2f with %s=%.4f", best_threshold, metric, best_score)
    return best_threshold


def _metric_score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    if metric == "f1":
        denom = 2 * tp + fp + fn
        return (2 * tp / denom) if denom > 0 else 0.0
    if metric == "precision":
        denom = tp + fp
        return (tp / denom) if denom > 0 else 0.0
    if metric == "recall":
        denom = tp + fn
        return (tp / denom) if denom > 0 else 0.0
    if metric == "accuracy":
        return (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    raise ValueError(f"Unknown metric: {metric!r}. Use 'f1', 'precision', 'recall', or 'accuracy'.")


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Convert probability scores to binary predictions using threshold."""
    return (np.asarray(scores) >= threshold).astype(int)


def threshold_summary(
    y_true: list[int] | np.ndarray,
    y_scores: list[float] | np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Return precision, recall, F1, and accuracy at the given threshold."""
    y_true_arr = np.asarray(y_true)
    y_scores_arr = np.asarray(y_scores)
    y_pred = (y_scores_arr >= threshold).astype(int)

    return {
        "threshold": round(threshold, 4),
        "precision": round(_metric_score(y_true_arr, y_pred, "precision"), 4),
        "recall": round(_metric_score(y_true_arr, y_pred, "recall"), 4),
        "f1": round(_metric_score(y_true_arr, y_pred, "f1"), 4),
        "accuracy": round(_metric_score(y_true_arr, y_pred, "accuracy"), 4),
    }
