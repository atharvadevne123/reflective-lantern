"""Batch scoring utilities for processing large applicant sets."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def chunk_list(items: list[Any], chunk_size: int) -> list[list[Any]]:
    """Split a list into chunks of at most chunk_size items."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def batch_score(
    applications: list[dict[str, Any]],
    model: Any,
    pipeline: Any,
    extract_fn: Any,
    predict_fn: Any,
    chunk_size: int = 128,
) -> list[float]:
    """Score a batch of applications in memory-efficient chunks.

    Returns list of default probabilities in the same order as input.
    """
    if not applications:
        return []

    all_scores: list[float] = []
    chunks = chunk_list(applications, chunk_size)

    for i, chunk in enumerate(chunks):
        logger.debug("Scoring chunk %d/%d (%d applications)", i + 1, len(chunks), len(chunk))
        try:
            features_list = [extract_fn(app) for app in chunk]
            X = np.vstack(features_list)
            scores = predict_fn(model, X, pipeline)
            all_scores.extend(scores.tolist())
        except Exception:
            logger.exception("Error scoring chunk %d; padding with NaN", i + 1)
            all_scores.extend([float("nan")] * len(chunk))

    return all_scores


def summarise_batch_results(scores: list[float]) -> dict[str, Any]:
    """Return aggregate statistics for a batch scoring run."""
    valid = [s for s in scores if not (s != s)]  # filter NaN
    total = len(scores)
    failed = total - len(valid)

    if not valid:
        return {"total": total, "failed": failed, "scored": 0}

    arr = np.array(valid)
    return {
        "total": total,
        "scored": len(valid),
        "failed": failed,
        "mean_score": float(round(arr.mean(), 4)),
        "median_score": float(round(float(np.median(arr)), 4)),
        "high_risk_count": int((arr >= 0.5).sum()),
        "high_risk_rate": float(round((arr >= 0.5).mean(), 4)),
    }
