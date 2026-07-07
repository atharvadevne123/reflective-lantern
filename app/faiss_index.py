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
    global _index
    if _index is None:
        _index = faiss.IndexFlatL2(DIM)
        logger.info("Initialised FAISS IndexFlatL2 dim=%d", DIM)
    return _index


def add_property(vector: np.ndarray, metadata: dict[str, Any]) -> None:
    vec = vector.reshape(1, -1).astype(np.float32)
    if vec.shape[1] < DIM:
        vec = np.pad(vec, ((0, 0), (0, DIM - vec.shape[1])))
    elif vec.shape[1] > DIM:
        vec = vec[:, :DIM]
    get_index().add(vec)
    _stored_properties.append(metadata)


def search_comparable(
    query_vector: np.ndarray, top_k: int = 5
) -> list[dict[str, Any]]:
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
    return get_index().ntotal


def reset_index() -> None:
    global _index, _stored_properties
    _index = faiss.IndexFlatL2(DIM)
    _stored_properties = []
