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
        if not self._vectors:
            return []
        q_norm = query / (np.linalg.norm(query) + 1e-9)
        sims = []
        for i, v in enumerate(self._vectors):
            v_norm = v / (np.linalg.norm(v) + 1e-9)
            sim = float(np.dot(q_norm, v_norm))
            sims.append((sim, i))
        sims.sort(reverse=True)
        return [
            {"rank": r + 1, "index": i, "similarity": round(s, 4), "metadata": self._metadata[i]}
            for r, (s, i) in enumerate(sims[:k])
        ]

    @property
    def size(self) -> int:
        return len(self._vectors)


_pattern_index: LoadPatternIndex | None = None


def get_pattern_index(dim: int = 24) -> LoadPatternIndex:
    global _pattern_index
    if _pattern_index is None:
        _pattern_index = LoadPatternIndex(dim=dim)
    return _pattern_index
