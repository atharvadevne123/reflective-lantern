"""Feature importance extraction from trained ensemble models."""

from __future__ import annotations

import numpy as np


def extract_feature_importances(
    pipeline,
    feature_names: list[str],
) -> list[tuple[str, float]]:
    """Return (name, importance) pairs sorted descending from an sklearn Pipeline.

    Args:
        pipeline: Fitted sklearn Pipeline whose last step supports feature_importances_
                  or coef_ (e.g., VotingClassifier, RandomForestClassifier).
        feature_names: Names aligned to the feature columns.

    Returns:
        List of (feature_name, importance) tuples, sorted by importance descending.
        Returns an empty list when importances cannot be extracted.
    """
    estimator = pipeline[-1] if hasattr(pipeline, "__getitem__") else pipeline
    importances: np.ndarray | None = None

    if hasattr(estimator, "feature_importances_"):
        importances = np.array(estimator.feature_importances_)
    elif hasattr(estimator, "estimators_"):
        # VotingClassifier — average importances across base learners that support it
        parts = []
        for _, est in estimator.estimators_:
            if hasattr(est, "feature_importances_"):
                parts.append(est.feature_importances_)
        if parts:
            importances = np.mean(parts, axis=0)
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_).ravel()

    if importances is None or len(importances) != len(feature_names):
        return []

    pairs = list(zip(feature_names, importances.tolist(), strict=False))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def top_k_features(
    pipeline,
    feature_names: list[str],
    k: int = 10,
) -> list[tuple[str, float]]:
    """Return the top-k most important features.

    Args:
        pipeline: Fitted sklearn Pipeline.
        feature_names: Feature names aligned to columns.
        k: Number of top features to return.

    Returns:
        Up to k (feature_name, importance) tuples.
    """
    all_pairs = extract_feature_importances(pipeline, feature_names)
    return all_pairs[:k]


def importance_as_dict(
    pipeline,
    feature_names: list[str],
) -> dict[str, float]:
    """Return feature importances as a name-to-score mapping.

    Args:
        pipeline: Fitted sklearn Pipeline.
        feature_names: Feature names aligned to columns.

    Returns:
        Dict mapping feature name to importance score, sorted descending.
    """
    pairs = extract_feature_importances(pipeline, feature_names)
    return {name: score for name, score in pairs}


def cumulative_importance(
    pipeline,
    feature_names: list[str],
    threshold: float = 0.90,
) -> list[str]:
    """Return the minimal feature set that explains ``threshold`` of total importance.

    Args:
        pipeline: Fitted sklearn Pipeline.
        feature_names: Feature names aligned to columns.
        threshold: Cumulative importance fraction to cover (default 0.90).

    Returns:
        Ordered list of feature names up to and including the threshold.
    """
    pairs = extract_feature_importances(pipeline, feature_names)
    if not pairs:
        return []
    total = sum(score for _, score in pairs)
    if total == 0:
        return []
    cumsum = 0.0
    selected: list[str] = []
    for name, score in pairs:
        cumsum += score / total
        selected.append(name)
        if cumsum >= threshold:
            break
    return selected
