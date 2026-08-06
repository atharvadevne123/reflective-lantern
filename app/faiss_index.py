"""FAISS-based similar load-period search for energy pattern matching."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore[import]

    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
    logger.info("faiss not available; using brute-force fallback")


class LoadPatternIndex:
    """Stores energy load vectors and finds similar historical periods."""

    def __init__(self, dim: int = 24) -> None:
        self.dim = dim
        self._vectors: list[np.ndarray] = []
        self._metadata: list[dict] = []
        self._index: object | None = None

    def add(self, vector: list[float], metadata: dict | None = None) -> None:
        """Add a load period vector (e.g. 24-hour profile) to the index."""
        v = np.array(vector, dtype=np.float32)
        if len(v) != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {len(v)}")
        self._vectors.append(v)
        self._metadata.append(metadata or {})
        self._index = None  # invalidate

    def build(self) -> None:
        """Build the FAISS index from stored vectors."""
        if not self._vectors:
            return
        if not _HAS_FAISS:
            logger.info("faiss unavailable — brute-force search will be used")
            return
        matrix = np.stack(self._vectors, axis=0).astype(np.float32)
        faiss.normalize_L2(matrix)
        idx = faiss.IndexFlatIP(self.dim)
        idx.add(matrix)
        self._index = idx
        logger.info("faiss index built with %d vectors", len(self._vectors))

    def search(self, query: list[float], k: int = 5) -> list[dict]:
        """Find k most similar load period profiles."""
        q = np.array(query, dtype=np.float32).reshape(1, -1)

        if _HAS_FAISS and self._index is not None:
            faiss.normalize_L2(q)
            distances, indices = self._index.search(q, k)
            return [
                {
                    "rank": r + 1,
                    "index": int(indices[0][r]),
                    "similarity": round(float(distances[0][r]), 4),
                    "metadata": self._metadata[int(indices[0][r])],
                }
                for r in range(min(k, len(self._vectors)))
                if indices[0][r] >= 0
            ]

        return self._brute_force_search(q[0], k)

    def _brute_force_search(self, query: np.ndarray, k: int) -> list[dict]:
        """Cosine-similarity search without FAISS (O(n) linear scan)."""
        if not self._vectors:
            return []
        q_norm = query / (np.linalg.norm(query) + 1e-9)
        sims: list[tuple[float, int]] = []
        for i, v in enumerate(self._vectors):
            v_norm = v / (np.linalg.norm(v) + 1e-9)
            sim = float(np.dot(q_norm, v_norm))
            sims.append((sim, i))
        sims.sort(reverse=True)
        return [
            {"rank": r + 1, "index": i, "similarity": round(s, 4), "metadata": self._metadata[i]}
            for r, (s, i) in enumerate(sims[:k])
        ]

    def clear(self) -> None:
        """Remove all vectors and invalidate the index."""
        self._vectors.clear()
        self._metadata.clear()
        self._index = None

    def reset(self) -> None:
        """Alias for clear(); remove all vectors and invalidate the index."""
        self.clear()

    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        return len(self._vectors)

    def save_vectors(self) -> np.ndarray:
        """Return a copy of all stored vectors as a 2-D float32 array.

        Returns:
            Shape (n, dim) array; empty (0, dim) when the index has no vectors.
        """
        if not self._vectors:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.stack(self._vectors, axis=0).copy()

    @property
    def size(self) -> int:
        """Number of vectors stored in the index."""
        return len(self._vectors)


_pattern_index: LoadPatternIndex | None = None

DIM: int = 24
DEFAULT_TOP_K: int = 5
MAX_TOP_K: int = 100

__all__ = [
    "DEFAULT_TOP_K",
    "DIM",
    "LoadPatternIndex",
    "MAX_TOP_K",
    "add_property",
    "get_pattern_index",
    "index_size",
    "reset_index",
    "search_comparable",
]


def get_pattern_index(dim: int = DIM) -> LoadPatternIndex:
    global _pattern_index
    if _pattern_index is None:
        _pattern_index = LoadPatternIndex(dim=dim)
    return _pattern_index


def add_property(vector: list[float] | np.ndarray, metadata: dict | None = None) -> None:
    """Add a property vector to the global comparable-property index.

    Args:
        vector: Feature vector; padded or truncated to DIM automatically.
        metadata: Optional dict of property attributes (e.g. zipcode, price).
    """
    v = np.array(vector, dtype=np.float32)
    if len(v) < DIM:
        v = np.pad(v, (0, DIM - len(v)))
    elif len(v) > DIM:
        v = v[:DIM]
    get_pattern_index().add(v.tolist(), metadata or {})


def index_size() -> int:
    """Return the number of property vectors stored in the global index."""
    return get_pattern_index().size


def reset_index() -> None:
    """Clear all vectors from the global comparable-property index."""
    get_pattern_index().clear()


def search_comparable(query: list[float] | np.ndarray, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Search the global index for the top-k most similar property vectors.

    Args:
        query: Feature vector; padded or truncated to DIM automatically.
        top_k: Maximum number of results to return (capped at MAX_TOP_K).

    Returns:
        List of dicts with 'distance' and any stored metadata keys (flat).
    """
    top_k = min(top_k, MAX_TOP_K)
    q = np.array(query, dtype=np.float32)
    if len(q) < DIM:
        q = np.pad(q, (0, DIM - len(q)))
    elif len(q) > DIM:
        q = q[:DIM]
    raw = get_pattern_index().search(q.tolist(), k=top_k)
    results = []
    for r in raw:
        entry = {"distance": round(1.0 - r.get("similarity", 0.0), 6)}
        entry.update(r.get("metadata", {}))
        results.append(entry)
    return results


def index_is_empty() -> bool:
    """Return True if the global FAISS index contains no vectors.

    Returns:
        True when index_size() == 0.
    """
    return index_size() == 0


def batch_add_properties(
    vectors: list[list[float]], metadata_list: list[dict | None] | None = None
) -> int:
    """Add multiple property vectors to the index in one call.

    Args:
        vectors: List of embedding vectors.
        metadata_list: Optional parallel list of metadata dicts. Defaults to all None.

    Returns:
        Number of vectors successfully added.
    """
    if metadata_list is None:
        metadata_list = [None] * len(vectors)
    added = 0
    for vec, meta in zip(vectors, metadata_list):
        try:
            add_property(vec, meta)
            added += 1
        except Exception:
            pass
    return added


def top_k_distances(query: list[float], top_k: int = 5) -> list[float]:
    """Return only the distance scores from a similarity search.

    Args:
        query: Query embedding vector.
        top_k: Number of results. Default 5.

    Returns:
        Sorted list of distance scores (ascending = more similar).
    """
    results = search_comparable(query, top_k=top_k)
    return [r["distance"] for r in results]


def index_summary() -> dict[str, object]:
    """Return a summary dict describing the current index state.

    Returns:
        Dict with 'size', 'is_empty', and 'dim' keys.
    """
    size = index_size()
    return {
        "size": size,
        "is_empty": size == 0,
        "dim": DIM,
    }
