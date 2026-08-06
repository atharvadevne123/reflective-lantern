"""FAISS-based anomaly root cause analysis via nearest-neighbour search."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore[import]

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not installed; nearest-neighbour search will use sklearn fallback")

_INDEX: Any = None
_INDEX_LABELS: list[dict[str, Any]] = []
_NN_MODEL: NearestNeighbors | None = None
_FEATURE_NAMES: list[str] = []


def build_index(
    X: np.ndarray,
    labels: list[dict[str, Any]],
    feature_names: list[str],
    n_lists: int = 100,
) -> None:
    """Build FAISS (or sklearn fallback) index from training embeddings."""
    global _INDEX, _INDEX_LABELS, _NN_MODEL, _FEATURE_NAMES

    _INDEX_LABELS = labels
    _FEATURE_NAMES = feature_names
    n, d = X.shape
    X_f32 = X.astype(np.float32)

    if FAISS_AVAILABLE:
        quantizer = faiss.IndexFlatL2(d)
        if n > n_lists * 10:
            index = faiss.IndexIVFFlat(quantizer, d, min(n_lists, n // 10))
            index.train(X_f32)
        else:
            index = quantizer
        index.add(X_f32)
        _INDEX = index
        logger.info("FAISS index built: %d vectors, dim=%d", n, d)
    else:
        _NN_MODEL = NearestNeighbors(n_neighbors=5, metric="euclidean", n_jobs=-1)
        _NN_MODEL.fit(X_f32)
        logger.info("sklearn NN index built: %d vectors, dim=%d", n, d)


def find_similar_anomalies(
    query: np.ndarray,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Return k nearest anomalies from the index with root cause hints.

    Compares the query vector to the stored reference anomalies and
    surfaces the features that diverge most from their neighbours.
    """
    if _INDEX is None and _NN_MODEL is None:
        return []

    query_f32 = query.astype(np.float32).reshape(1, -1)

    if FAISS_AVAILABLE and _INDEX is not None:
        distances, indices = _INDEX.search(query_f32, min(k, len(_INDEX_LABELS)))
        distances = distances[0]
        indices = indices[0]
    else:
        assert _NN_MODEL is not None
        distances_arr, indices_arr = _NN_MODEL.kneighbors(
            query_f32, n_neighbors=min(k, len(_INDEX_LABELS))
        )
        distances = distances_arr[0]
        indices = indices_arr[0]

    results = []
    for dist, idx in zip(distances, indices):
        if idx < 0 or idx >= len(_INDEX_LABELS):
            continue
        label = _INDEX_LABELS[idx].copy()
        label["distance"] = float(dist)
        results.append(label)
    return results


def explain_anomaly(
    query_vector: np.ndarray,
    feature_names: list[str] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Identify which features contribute most to the anomaly score.

    Uses absolute z-score deviation from the feature mean as the
    contribution signal, suitable when no reference index exists.
    """
    fn = feature_names or _FEATURE_NAMES or [f"f{i}" for i in range(len(query_vector))]
    z_scores = np.abs(query_vector - np.mean(query_vector)) / (np.std(query_vector) + 1e-9)
    top_indices = np.argsort(z_scores)[::-1][:top_k]
    contributors = [
        {"feature": fn[i] if i < len(fn) else f"f{i}", "z_score": float(z_scores[i])}
        for i in top_indices
    ]
    similar = find_similar_anomalies(query_vector, k=3)
    return {"top_contributors": contributors, "similar_historical_anomalies": similar}
