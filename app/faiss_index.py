"""FAISS-based comparable property search.

Maintains a module-level IndexFlatL2 (L2 distance brute-force) alongside
a list of metadata dicts so callers can retrieve full context for each
nearest neighbour.

All vectors are padded or truncated to DIM=24 to match the feature space.
The index is rebuilt fresh on each process restart; for production workloads
consider persisting the index with faiss.write_index / faiss.read_index.
"""

import logging
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)

_index: faiss.IndexFlatL2 | None = None
_stored_properties: list[dict[str, Any]] = []
DIM = 24


def get_index() -> faiss.IndexFlatL2:
    """Return the module-level FAISS index, creating it on first call."""
    global _index
    if _index is None:
        _index = faiss.IndexFlatL2(DIM)
        logger.info("Initialised FAISS IndexFlatL2 dim=%d", DIM)
    return _index


def add_property(vector: np.ndarray, metadata: dict[str, Any]) -> None:
    """Add a property feature vector and its metadata to the FAISS index.

    Args:
        vector: 1-D or 2-D numpy array of feature values; padded/truncated to DIM.
        metadata: Arbitrary dict stored alongside the vector for retrieval.
    """
    vec = vector.reshape(1, -1).astype(np.float32)
    if vec.shape[1] < DIM:
        vec = np.pad(vec, ((0, 0), (0, DIM - vec.shape[1])))
    elif vec.shape[1] > DIM:
        vec = vec[:, :DIM]
    get_index().add(vec)
    _stored_properties.append(metadata)


def search_comparable(query_vector: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
    """Return the *top_k* most similar properties from the FAISS index.

    Args:
        query_vector: Feature vector for the property being queried.
        top_k: Maximum number of neighbours to return.

    Returns:
        List of metadata dicts enriched with a ``distance`` key (L2 distance).
        Empty list if the index has no entries.
    """
    index = get_index()
    if index.ntotal == 0:
        return []
    vec = query_vector.reshape(1, -1).astype(np.float32)
    if vec.shape[1] < DIM:
        vec = np.pad(vec, ((0, 0), (0, DIM - vec.shape[1])))
    elif vec.shape[1] > DIM:
        vec = vec[:, :DIM]
    k = min(top_k, index.ntotal)
    distances, indices = index.search(vec, k)
    results = []
    for dist, idx in zip(distances[0], indices[0], strict=False):
        if idx < len(_stored_properties):
            entry = dict(_stored_properties[idx])
            entry["distance"] = float(dist)
            results.append(entry)
    return results


def index_size() -> int:
    """Return the number of vectors currently stored in the FAISS index."""
    return get_index().ntotal


def reset_index() -> None:
    """Clear all vectors and metadata from the module-level FAISS index."""
    global _index, _stored_properties
    _index = faiss.IndexFlatL2(DIM)
    _stored_properties = []
