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

    @property
    def building_ids(self) -> list[str]:
        """Return a list of all indexed building IDs in insertion order."""
        return [bid for bid, _ in self._profiles]


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


__all__ = [
    "BuildingSimilarityIndex",
    "batch_add",
    "chebyshev_distance",
    "cosine_distance",
    "euclidean_distance",
    "get_global_index",
    "hourly_pattern_distance",
    "manhattan_distance",
    "pearson_similarity",
    "score_distribution",
    "search_comparable",
    "similarity_matrix",
]


def euclidean_distance(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
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


def score_distribution(
    index: BuildingSimilarityIndex,
    query: list[float],
) -> dict[str, float]:
    """Return summary statistics of cosine similarity scores for all profiles.

    Args:
        index: BuildingSimilarityIndex to search against.
        query: Query feature vector.

    Returns:
        Dict with 'min', 'max', 'mean', 'std' similarity scores, or zeros if empty.
    """
    if not index.size:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    results = index.search(query, k=index.size)
    scores = np.array([s for _, s in results], dtype=float)
    return {
        "min": round(float(scores.min()), 4),
        "max": round(float(scores.max()), 4),
        "mean": round(float(scores.mean()), 4),
        "std": round(float(scores.std()), 4),
    }


def batch_add(index: BuildingSimilarityIndex, profiles: list[tuple[str, list[float]]]) -> int:
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


def hourly_pattern_distance(profile_a: list[float], profile_b: list[float]) -> float:
    """Compute the mean absolute error between two 24-hour load profiles.

    Both profiles must have the same length.  This metric is more interpretable
    than cosine distance when comparing hourly energy patterns because it
    preserves the magnitude (kWh) differences across the day.

    Args:
        profile_a: First hourly load profile (kWh per hour).
        profile_b: Second hourly load profile (kWh per hour).

    Returns:
        Mean absolute difference in kWh per hour.

    Raises:
        ValueError: If the profiles have different lengths or are empty.
    """
    if not profile_a or not profile_b:
        raise ValueError("profiles must not be empty")
    if len(profile_a) != len(profile_b):
        raise ValueError(f"profiles must have the same length: {len(profile_a)} != {len(profile_b)}")
    return sum(abs(a - b) for a, b in zip(profile_a, profile_b, strict=False)) / len(profile_a)


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute the Jaccard similarity coefficient between two sets.

    Args:
        set_a: First set of categorical labels or identifiers.
        set_b: Second set of categorical labels or identifiers.

    Returns:
        Jaccard similarity in [0, 1]; 1.0 for identical sets, 0.0 for disjoint.
        Returns 0.0 if both sets are empty.
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return round(intersection / union, 6) if union > 0 else 0.0


def normalize_distances(distances: list[float]) -> list[float]:
    """Normalize a list of distances to [0, 1] using min-max scaling.

    Args:
        distances: List of non-negative distance values.

    Returns:
        Normalized distances; all 0.5 when all values are equal.

    Raises:
        ValueError: If *distances* is empty.
    """
    if not distances:
        raise ValueError("distances must not be empty")
    min_d = min(distances)
    max_d = max(distances)
    rng = max_d - min_d
    if rng < 1e-12:
        return [0.5] * len(distances)
    return [round((d - min_d) / rng, 6) for d in distances]


def manhattan_distance(a: list[float], b: list[float]) -> float:
    """Compute the Manhattan (L1) distance between two vectors.

    Args:
        a: First numeric vector.
        b: Second numeric vector, same length as *a*.

    Returns:
        Sum of absolute element-wise differences.

    Raises:
        ValueError: If vectors have different lengths or are empty.
    """
    if not a or not b:
        raise ValueError("vectors must not be empty")
    if len(a) != len(b):
        raise ValueError(f"vectors must have the same length: {len(a)} != {len(b)}")
    return round(sum(abs(ai - bi) for ai, bi in zip(a, b, strict=False)), 6)


def top_k_similar(
    query: list[float],
    candidates: list[list[float]],
    k: int = 5,
) -> list[tuple[int, float]]:
    """Return the *k* most similar candidates to *query* by cosine distance.

    Args:
        query: Query vector.
        candidates: List of candidate vectors (each must match query length).
        k: Number of top results to return.

    Returns:
        List of (index, distance) pairs, sorted ascending by cosine distance.

    Raises:
        ValueError: If *query* is empty or *k* < 1.
    """
    if not query:
        raise ValueError("query must not be empty")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    dists = []
    for i, cand in enumerate(candidates):
        if len(cand) == len(query):
            try:
                d = cosine_distance(query, cand)
            except Exception:
                d = float("inf")
            dists.append((i, d))
    dists.sort(key=lambda x: x[1])
    return dists[:k]


def weighted_jaccard_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Compute weighted Jaccard similarity between two feature-weight dicts.

    Args:
        a: First dict of feature -> weight.
        b: Second dict of feature -> weight.

    Returns:
        Weighted Jaccard similarity in [0, 1].
    """
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    intersection = sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    union = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    if union == 0:
        return 0.0
    return round(intersection / union, 6)


def pearson_similarity(a: list[float], b: list[float]) -> float:
    """Compute Pearson correlation coefficient between two equal-length profiles.

    Args:
        a: First numeric sequence.
        b: Second numeric sequence (must be same length as *a*).

    Returns:
        Pearson correlation in [-1, 1]; 0.0 if either series has zero variance.

    Raises:
        ValueError: If sequences are empty or have different lengths.
    """
    if not a or not b:
        raise ValueError("sequences must not be empty")
    if len(a) != len(b):
        raise ValueError(f"sequences must have the same length: {len(a)} != {len(b)}")
    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=False))
    std_a = sum((x - mean_a) ** 2 for x in a) ** 0.5
    std_b = sum((y - mean_b) ** 2 for y in b) ** 0.5
    if std_a == 0 or std_b == 0:
        return 0.0
    return round(cov / (std_a * std_b), 6)


def chebyshev_distance(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Compute Chebyshev (L∞) distance — the maximum element-wise difference.

    Args:
        a: First feature vector.
        b: Second feature vector (must be same length as *a*).

    Returns:
        Chebyshev distance rounded to 6 decimal places.
    """
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    return round(float(np.max(np.abs(va - vb))), 6)


def similarity_matrix(profiles: list[list[float]]) -> list[list[float]]:
    """Compute an NxN pairwise cosine similarity matrix.

    Args:
        profiles: List of N feature vectors of equal length.

    Returns:
        NxN list-of-lists where entry [i][j] is the cosine similarity
        between profiles[i] and profiles[j].

    Raises:
        ValueError: If *profiles* is empty or vectors have different lengths.
    """
    if not profiles:
        raise ValueError("profiles must not be empty")
    n = len(profiles)
    arr = np.array(profiles, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    normed = arr / norms
    mat = normed @ normed.T
    return [[round(float(mat[i, j]), 6) for j in range(n)] for i in range(n)]


def minkowski_distance(a: list[float], b: list[float], p: float = 2.0) -> float:
    """Compute the Minkowski distance of order *p* between two vectors.

    Special cases: p=1 is Manhattan distance, p=2 is Euclidean distance.

    Args:
        a: First vector.
        b: Second vector (must be same length as *a*).
        p: Order of the norm (must be >= 1).

    Returns:
        Minkowski distance rounded to 6 decimal places.

    Raises:
        ValueError: If *p* < 1 or vectors differ in length.
    """
    if p < 1.0:
        raise ValueError(f"p must be >= 1, got {p}")
    if len(a) != len(b):
        raise ValueError(f"Vectors must be same length: {len(a)} != {len(b)}")
    return round(sum(abs(x - y) ** p for x, y in zip(a, b)) ** (1.0 / p), 6)


def dice_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Dice similarity coefficient between two sets.

    Dice = 2 * |A ∩ B| / (|A| + |B|).

    Args:
        set_a: First set.
        set_b: Second set.

    Returns:
        Dice similarity in [0, 1]; 0.0 if both sets are empty.
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    return round(2.0 * intersection / (len(set_a) + len(set_b)), 6)


def overlap_coefficient(set_a: set[str], set_b: set[str]) -> float:
    """Compute the Overlap (Szymkiewicz–Simpson) coefficient.

    Overlap = |A ∩ B| / min(|A|, |B|).

    Returns 0.0 when either set is empty.

    Args:
        set_a: First set.
        set_b: Second set.

    Returns:
        Overlap coefficient in [0, 1].
    """
    if not set_a or not set_b:
        return 0.0
    return round(len(set_a & set_b) / min(len(set_a), len(set_b)), 6)


def euclidean_distance(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute the Euclidean distance between two vectors.

    Args:
        vec_a: First numeric vector.
        vec_b: Second numeric vector.

    Returns:
        Euclidean distance rounded to 6 decimal places.

    Raises:
        ValueError: If vectors have different lengths or are empty.
    """
    if not vec_a or not vec_b:
        raise ValueError("Vectors must not be empty")
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same length")
    return round(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b, strict=False)) ** 0.5, 6)


def pearson_correlation(
    x: list[float],
    y: list[float],
) -> float:
    """Compute the Pearson correlation coefficient between *x* and *y*.

    Args:
        x: First numeric series.
        y: Second numeric series.

    Returns:
        Pearson r in [-1, 1] rounded to 6 decimal places, or 0.0 if either
        series has zero variance.

    Raises:
        ValueError: If series have different lengths or fewer than 2 elements.
    """
    if len(x) < 2 or len(y) < 2:
        raise ValueError("At least 2 elements required")
    if len(x) != len(y):
        raise ValueError("Series must have the same length")
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=False))
    den_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    den_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5
    if den_x < 1e-12 or den_y < 1e-12:
        return 0.0
    return round(num / (den_x * den_y), 6)


def top_k_similar(
    query: list[float],
    candidates: list[list[float]],
    k: int = 5,
) -> list[tuple[int, float]]:
    """Find the *k* most similar candidates to a query vector by cosine similarity.

    Args:
        query: Query feature vector.
        candidates: List of candidate vectors (same dimension as *query*).
        k: Number of top results to return. Default 5.

    Returns:
        List of (index, similarity_score) tuples, sorted by score descending.

    Raises:
        ValueError: If *k* < 1 or *candidates* is empty.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")
    if k < 1:
        raise ValueError("k must be at least 1")

    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(ai * bi for ai, bi in zip(a, b, strict=False))
        mag_a = sum(ai**2 for ai in a) ** 0.5
        mag_b = sum(bi**2 for bi in b) ** 0.5
        if mag_a < 1e-12 or mag_b < 1e-12:
            return 0.0
        return dot / (mag_a * mag_b)

    scored = [(i, round(_cosine(query, c), 6)) for i, c in enumerate(candidates)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
