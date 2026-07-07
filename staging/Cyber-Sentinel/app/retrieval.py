"""FAISS-based pattern matching for known attack signatures."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 25
_index_cache: dict[str, Any] = {}


class AttackPatternIndex:
    """Cosine-similarity index over known attack signature embeddings.

    Falls back to brute-force numpy search if faiss is unavailable.

    Args:
        dim: Feature dimension (must match build_feature_vector output length).
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim
        self._embeddings: list[np.ndarray] = []
        self._labels: list[str] = []
        self._use_faiss = False

        try:
            import faiss

            self._index = faiss.IndexFlatIP(dim)
            self._use_faiss = True
            logger.info("FAISS index initialised (dim=%d)", dim)
        except ImportError:
            logger.warning("faiss not available; using numpy brute-force search")
            self._index = None

        self._seed_known_patterns()

    def _seed_known_patterns(self) -> None:
        """Add a small set of synthetic attack signature embeddings."""
        rng = np.random.default_rng(0)
        attack_types = ["dos", "probe", "r2l", "u2r"]
        for i, attack in enumerate(attack_types):
            for _ in range(10):
                vec = rng.normal(loc=float(i), scale=0.3, size=self.dim).astype(np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                self._embeddings.append(vec)
                self._labels.append(attack)

        if self._use_faiss and self._embeddings:
            matrix = np.vstack(self._embeddings)
            self._index.add(matrix)

    def search(self, query: list[float] | np.ndarray, k: int = 3) -> list[dict[str, Any]]:
        """Find the k most similar known attack patterns.

        Args:
            query: Feature vector of the incoming event.
            k: Number of nearest neighbours to return.

        Returns:
            List of dicts with rank, similarity, and attack_type.
        """
        q = np.asarray(query, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        if self._use_faiss and self._index is not None:
            q_2d = q.reshape(1, -1)
            distances, indices = self._index.search(q_2d, min(k, len(self._embeddings)))
            return [
                {
                    "rank": rank + 1,
                    "similarity": round(float(distances[0][rank]), 4),
                    "attack_type": self._labels[indices[0][rank]],
                }
                for rank in range(len(indices[0]))
                if indices[0][rank] >= 0
            ]

        if not self._embeddings:
            return []

        matrix = np.vstack(self._embeddings)
        similarities = matrix @ q
        top_k = int(min(k, len(similarities)))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [
            {
                "rank": rank + 1,
                "similarity": round(float(similarities[idx]), 4),
                "attack_type": self._labels[idx],
            }
            for rank, idx in enumerate(top_indices)
        ]

    @property
    def size(self) -> int:
        """Number of patterns stored in the index."""
        return len(self._embeddings)


def get_index(dim: int = EMBEDDING_DIM) -> AttackPatternIndex:
    """Return singleton AttackPatternIndex for the given dimension.

    Args:
        dim: Feature dimension. Changing this creates a new index.

    Returns:
        Cached or freshly created AttackPatternIndex.
    """
    cache_key = f"index_{dim}"
    if cache_key not in _index_cache:
        _index_cache[cache_key] = AttackPatternIndex(dim=dim)
        logger.info("Created new AttackPatternIndex(dim=%d)", dim)
    return _index_cache[cache_key]
