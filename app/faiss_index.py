"""FAISS-based comparable property search.

Maintains a module-level index alongside a list of metadata dicts so callers
can retrieve full context for each nearest neighbour.

All vectors are padded or truncated to DIM=24 to match the feature space.
The index is rebuilt fresh on each process restart; for production workloads
consider persisting the index with faiss.write_index / faiss.read_index.

When the ``faiss`` package is unavailable, a brute-force NumPy fallback is
used transparently so the module remains importable in all environments.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss as _faiss

    _HAS_FAISS = True
    logger.debug("faiss available — using IndexFlatL2")
except ImportError:
    _faiss = None  # type: ignore[assignment]
    _HAS_FAISS = False
    logger.info("faiss not available; using brute-force NumPy fallback")

DIM = 24
DEFAULT_TOP_K = 5
MAX_TOP_K = 50

_index: Any = None
_stored_properties: list[dict[str, Any]] = []
_vectors: list[np.ndarray] = []


def _normalize(vector: np.ndarray) -> np.ndarray:
    """Pad or truncate *vector* to shape (1, DIM) as float32."""
    vec = vector.reshape(1, -1).astype(np.float32)
    if vec.shape[1] < DIM:
        vec = np.pad(vec, ((0, 0), (0, DIM - vec.shape[1])))
    elif vec.shape[1] > DIM:
        vec = vec[:, :DIM]
    return vec


def get_index() -> Any:
    """Return the module-level index, creating it on first call."""
    global _index
    if _index is None and _HAS_FAISS:
        _index = _faiss.IndexFlatL2(DIM)
        logger.info("Initialised FAISS IndexFlatL2 dim=%d", DIM)
    return _index


def add_property(vector: np.ndarray, metadata: dict[str, Any]) -> None:
    """Add a property feature vector and its metadata to the index.

    Args:
        vector: 1-D or 2-D numpy array of feature values; padded/truncated to DIM.
        metadata: Arbitrary dict stored alongside the vector for retrieval.
    """
    vec = _normalize(vector)
    if _HAS_FAISS:
        get_index().add(vec)
    _vectors.append(vec[0])
    _stored_properties.append(metadata)


def search_comparable(query_vector: np.ndarray, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    """Return the *top_k* most similar properties from the index.

    Args:
        query_vector: Feature vector for the property being queried.
        top_k: Maximum number of neighbours to return.

    Returns:
        List of metadata dicts enriched with a ``distance`` key (L2 distance).
        Empty list if the index has no entries.
    """
    if not _stored_properties:
        return []
    vec = _normalize(query_vector)
    k = min(top_k, len(_stored_properties))

    if _HAS_FAISS and _index is not None:
        distances, indices = _index.search(vec, k)
        raw_dists = distances[0]
        raw_idxs = indices[0]
    else:
        matrix = np.stack(_vectors, axis=0)
        diffs = matrix - vec
        raw_dists_all = np.sum(diffs**2, axis=1)
        raw_idxs = np.argsort(raw_dists_all)[:k]
        raw_dists = raw_dists_all[raw_idxs]

    results = []
    for dist, idx in zip(raw_dists, raw_idxs, strict=False):
        if 0 <= idx < len(_stored_properties):
            entry = dict(_stored_properties[idx])
            entry["distance"] = float(dist)
            results.append(entry)
    return results


def index_size() -> int:
    """Return the number of vectors currently stored in the index."""
    return len(_stored_properties)


def reset_index() -> None:
    """Clear all vectors and metadata from the module-level index."""
    global _index, _stored_properties, _vectors
    if _HAS_FAISS:
        _index = _faiss.IndexFlatL2(DIM)
    else:
        _index = None
    _stored_properties = []
    _vectors = []
