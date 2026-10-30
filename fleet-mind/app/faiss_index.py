"""FAISS-based nearest-neighbour index for anomaly similarity search.

When faiss-cpu is installed, builds and queries a real FAISS index.
Falls back to a brute-force numpy implementation when faiss is absent.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore[import]
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

_index: Any = None
_stored_ids: list[str] = []


def build_index(vectors: np.ndarray, vehicle_ids: list[str]) -> None:
    """Build (or rebuild) the nearest-neighbour index from training vectors.

    Args:
        vectors: 2-D float32 array of shape (n_samples, n_features).
        vehicle_ids: Parallel list of vehicle IDs corresponding to each row.
    """
    global _index, _stored_ids
    _stored_ids = list(vehicle_ids)

    if _HAS_FAISS:
        dim = vectors.shape[1]
        _index = faiss.IndexFlatL2(dim)
        _index.add(vectors.astype(np.float32))
        logger.info("FAISS index built: %d vectors (dim=%d)", len(vehicle_ids), dim)
    else:
        _index = vectors.astype(np.float32)
        logger.info("Brute-force index built: %d vectors (dim=%d)", len(vehicle_ids), vectors.shape[1])


def query_similar(
    query_vector: np.ndarray,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Return the k most similar vehicles to the query vector.

    Args:
        query_vector: 1-D or 2-D float array representing the query telemetry.
        k: Number of nearest neighbours to return.

    Returns:
        List of dicts with 'vehicle_id' and 'distance' for each neighbour.
        Empty list if the index has not been built yet.
    """
    if _index is None:
        return []

    q = np.atleast_2d(query_vector).astype(np.float32)
    k_actual = min(k, len(_stored_ids))

    if _HAS_FAISS:
        distances, indices = _index.search(q, k_actual)
        results = [
            {"vehicle_id": _stored_ids[int(idx)], "distance": float(dist)}
            for dist, idx in zip(distances[0], indices[0])
            if idx >= 0
        ]
    else:
        diffs = _index - q
        dists = np.linalg.norm(diffs, axis=1)
        top_k = np.argsort(dists)[:k_actual]
        results = [
            {"vehicle_id": _stored_ids[int(i)], "distance": float(dists[i])}
            for i in top_k
        ]

    return results
