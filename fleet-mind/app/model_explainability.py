"""Feature importance and basic model explainability utilities."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.features import ALL_FEATURES

logger = logging.getLogger(__name__)


def get_feature_importances(model: Any) -> list[dict[str, Any]]:
    """Extract per-feature importance scores from the trained ensemble.

    Works with XGBoost, LightGBM, Sklearn VotingClassifier, and any estimator
    that exposes ``feature_importances_``.

    Args:
        model: Trained model or VotingClassifier instance.

    Returns:
        List of dicts sorted by importance descending, each with 'feature' and 'importance'.
        Returns an empty list if importances cannot be extracted.
    """
    importances: np.ndarray | None = None

    if hasattr(model, "estimators_"):
        # VotingClassifier — average across sub-estimators
        imp_list = []
        for _, estimator in model.estimators_:
            if hasattr(estimator, "feature_importances_"):
                imp_list.append(estimator.feature_importances_)
        if imp_list:
            importances = np.mean(imp_list, axis=0)
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    if importances is None:
        logger.warning("Model does not expose feature_importances_")
        return []

    if len(importances) != len(ALL_FEATURES):
        logger.warning(
            "Feature count mismatch: model has %d importances, expected %d",
            len(importances),
            len(ALL_FEATURES),
        )
        return []

    results = [
        {"feature": name, "importance": round(float(imp), 6)}
        for name, imp in zip(ALL_FEATURES, importances)
    ]
    results.sort(key=lambda x: x["importance"], reverse=True)
    return results


def top_contributing_features(
    model: Any,
    n: int = 5,
) -> list[dict[str, Any]]:
    """Return the top-N most important features.

    Args:
        model: Trained model instance.
        n: Number of top features to return.

    Returns:
        List of up to n dicts with 'feature' and 'importance'.
    """
    all_importances = get_feature_importances(model)
    return all_importances[:n]


def importance_above_threshold(
    model: Any,
    threshold: float = 0.05,
) -> list[dict[str, Any]]:
    """Return features whose importance exceeds a given threshold.

    Args:
        model: Trained model instance.
        threshold: Minimum importance (inclusive) to include.

    Returns:
        Filtered and sorted list of importance dicts.
    """
    return [f for f in get_feature_importances(model) if f["importance"] >= threshold]


def cumulative_importance(
    model: Any,
    target: float = 0.9,
) -> list[dict[str, Any]]:
    """Return the smallest set of features whose cumulative importance >= target.

    Args:
        model: Trained model instance.
        target: Cumulative importance threshold (0–1).

    Returns:
        List of importance dicts covering at least `target` of total importance.
    """
    ranked = get_feature_importances(model)
    total = sum(f["importance"] for f in ranked)
    if total == 0:
        return ranked
    result, running = [], 0.0
    for entry in ranked:
        result.append(entry)
        running += entry["importance"] / total
        if running >= target:
            break
    return result
