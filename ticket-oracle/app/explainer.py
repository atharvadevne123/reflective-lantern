"""Lightweight feature-importance explainer for Ticket-Oracle predictions.

Provides top-N feature contributions via a coefficient/importance
lookup so the API can surface human-readable rationale without
the overhead of SHAP on every inference.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def extract_feature_names(pipeline) -> list[str]:
    """Extract ordered feature names from a fitted ColumnTransformer pipeline.

    Args:
        pipeline: A fitted sklearn Pipeline whose first step is a ColumnTransformer.

    Returns:
        List of feature name strings in transformer output order.
    """
    try:
        ct = pipeline.named_steps["features"]
        return list(ct.get_feature_names_out())
    except Exception:
        return []


def top_features(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Return the top-N features by absolute importance.

    Args:
        feature_names: Names matching importances array length.
        importances: 1-D array of raw importance / coefficient values.
        top_n: Number of top features to return.

    Returns:
        List of dicts with 'feature' and 'importance' (absolute value, rounded).
    """
    if len(feature_names) != len(importances):
        logger.warning("feature_name_mismatch", extra={"names": len(feature_names), "importances": len(importances)})
        return []

    abs_imp = np.abs(importances)
    top_idx = np.argsort(abs_imp)[::-1][:top_n]
    return [
        {"feature": feature_names[i], "importance": round(float(abs_imp[i]), 6)}
        for i in top_idx
    ]


def explain_prediction(
    pipeline,
    feature_names: list[str],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Extract top feature importances from a fitted classifier pipeline.

    Works with tree-based estimators exposing feature_importances_ and
    linear estimators exposing coef_. Returns empty list for unsupported types.

    Args:
        pipeline: Fitted sklearn Pipeline with a classifier as its last step.
        feature_names: Feature names in transformer output order.
        top_n: How many top features to include.

    Returns:
        List of {feature, importance} dicts sorted by absolute importance.
    """
    try:
        clf = pipeline.named_steps.get("clf") or list(pipeline.named_steps.values())[-1]
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            coef = clf.coef_
            importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef[0])
        else:
            return []
        return top_features(feature_names, importances, top_n)
    except Exception as exc:
        logger.warning("explain_prediction_failed", extra={"error": str(exc)})
        return []
