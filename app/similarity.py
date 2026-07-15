"""Building energy profile similarity search using brute-force cosine distance."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class BuildingSimilarityIndex:
    """In-memory nearest-neighbour index for building energy profiles.

    Falls back to a brute-force cosine search when FAISS is unavailable.
    """

    def __init__(self) -> None:
        self._profiles: list[tuple[str, np.ndarray]] = []
        self._use_faiss = False
        try:
            import faiss  # noqa: F401

            self._use_faiss = True
            logger.info("FAISS available — using approximate search.")
        except ImportError:
            logger.info("FAISS not installed — using brute-force cosine search.")

    def add(self, building_id: str, profile: list[float]) -> None:
        """Add a building energy profile vector to the index.

        Args:
            building_id: Unique identifier for the building.
            profile: Feature vector representing the building's energy signature.
        """
        self._profiles.append((building_id, np.array(profile, dtype=np.float32)))

    def search(self, query: list[float], k: int = 5) -> list[tuple[str, float]]:
        """Return the k most similar buildings by cosine similarity.

        Args:
            query: Feature vector to search for.
            k: Maximum number of results to return.

        Returns:
            List of (building_id, similarity_score) pairs, highest score first.
        """
        if not self._profiles:
            return []
        q = np.array(query, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-9)
        scores: list[tuple[str, float]] = []
        for bid, vec in self._profiles:
            v_norm = vec / (np.linalg.norm(vec) + 1e-9)
            sim = float(np.dot(q_norm, v_norm))
            scores.append((bid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def clear(self) -> None:
        """Remove all profiles from the index."""
        self._profiles.clear()

    def __len__(self) -> int:
        """Return the number of profiles stored in the index."""
        return len(self._profiles)

    @property
    def size(self) -> int:
        """Alias for ``len(self)``."""
        return len(self._profiles)


_global_index = BuildingSimilarityIndex()


def get_global_index() -> BuildingSimilarityIndex:
    """Return the module-level singleton similarity index."""
    return _global_index


def search_comparable(query_vec: list[float] | np.ndarray, top_k: int = 5) -> list[dict[str, object]]:
    """Search the global index for comparable properties.

    Args:
        query_vec: Feature vector for the query property.
        top_k: Maximum number of comparable properties to return.

    Returns:
        List of dicts with 'building_id' and 'similarity_score'.
    """
    results = _global_index.search(list(query_vec), k=top_k)
    return [{"building_id": bid, "similarity_score": round(score, 4)} for bid, score in results]


def cosine_distance(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Compute 1 - cosine_similarity between two vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    sim = float(np.dot(va, vb) / ((np.linalg.norm(va) + 1e-9) * (np.linalg.norm(vb) + 1e-9)))
    return round(1.0 - sim, 6)


__all__ = ["BuildingSimilarityIndex", "cosine_distance", "get_global_index", "search_comparable"]


def euclidean_distance(a: "list[float] | np.ndarray", b: "list[float] | np.ndarray") -> float:
    """Compute Euclidean (L2) distance between two vectors.

    Args:
        a: First feature vector.
        b: Second feature vector (must be same length as *a*).

    Returns:
        L2 distance rounded to 6 decimal places.
    """
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    return round(float(np.linalg.norm(va - vb)), 6)


def batch_add(index: BuildingSimilarityIndex, profiles: "list[tuple[str, list[float]]]") -> int:
    """Add multiple building profiles to *index* in one call.

    Args:
        index: Target BuildingSimilarityIndex instance.
        profiles: List of (building_id, feature_vector) tuples.

    Returns:
        Number of profiles successfully added.
    """
    added = 0
    for building_id, profile in profiles:
        try:
            index.add(building_id, profile)
            added += 1
        except Exception:
            logger.exception("Failed to add profile for building %s", building_id)
    return added
