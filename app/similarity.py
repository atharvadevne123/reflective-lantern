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
    "batch_similarity_matrix",
    "chebyshev_distance",
    "cosine_distance",
    "dice_similarity",
    "euclidean_distance",
    "get_global_index",
    "hourly_pattern_distance",
    "jaccard_similarity",
    "manhattan_distance",
    "minkowski_distance",
    "normalize_distances",
    "normalize_profile",
    "overlap_coefficient",
    "pearson_correlation",
    "pearson_similarity",
    "score_distribution",
    "search_comparable",
    "similarity_matrix",
    "top_k_similar",
    "weighted_jaccard_similarity",
]


def euclidean_distance(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Compute Euclidean (L2) distance between two vectors.

    Args:
        a: First feature vector.
        b: Second feature vector (must be same length as *a*).

    Returns:
        L2 distance rounded to 6 decimal places.

    Raises:
        ValueError: If either vector is empty or their lengths differ.
    """
    if len(a) == 0 or len(b) == 0:
        raise ValueError("vectors must be non-empty")
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
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
        return [0.0] * len(distances)
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
    if not candidates:
        raise ValueError("candidates must not be empty")
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


def normalize_profile(profile: list[float]) -> list[float]:
    """Return L2-normalized version of *profile*.

    Args:
        profile: Raw feature vector as a list of floats.

    Returns:
        Unit-length vector (L2 norm = 1.0) as a list of floats.
        Returns the zero vector unchanged if the norm is zero.
    """
    arr = np.array(profile, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return list(arr)
    return [round(float(x), 8) for x in arr / norm]


def batch_similarity_matrix(profiles: list[list[float]]) -> list[list[float]]:
    """Compute an NxN pairwise cosine similarity matrix for *profiles*.

    Args:
        profiles: List of feature vectors (all must have the same length).

    Returns:
        NxN list-of-lists where entry [i][j] is the cosine similarity
        between profile i and profile j.  Values are rounded to 6 decimals.
        Returns an empty list for empty input.

    Raises:
        ValueError: If profiles have inconsistent lengths.
    """
    if not profiles:
        return []
    dim = len(profiles[0])
    if any(len(p) != dim for p in profiles):
        raise ValueError("All profiles must have the same length")
    arr = np.array(profiles, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    normed = arr / norms
    matrix = normed @ normed.T
    n = len(profiles)
    return [[round(float(matrix[i, j]), 6) for j in range(n)] for i in range(n)]


def minkowski_distance(a: list[float], b: list[float], p: float = 2.0) -> float:
    """Return the Minkowski distance of order *p* between two vectors.

    Args:
        a: First vector.
        b: Second vector (same length as *a*).
        p: Order of the norm (must be ≥ 1). ``p=1`` is Manhattan, ``p=2`` is Euclidean.

    Returns:
        Non-negative distance.

    Raises:
        ValueError: If lengths differ or *p* < 1.
    """
    if len(a) != len(b):
        raise ValueError(f"vectors must be same length, got {len(a)} and {len(b)}")
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    total = sum(abs(x - y) ** p for x, y in zip(a, b, strict=False))
    return round(total ** (1.0 / p), 6)


def dice_similarity(a: set, b: set) -> float:
    """Return the Sørensen-Dice coefficient between two sets.

    Args:
        a: First set.
        b: Second set.

    Returns:
        ``2 * |a ∩ b| / (|a| + |b|)``, or 0.0 when both sets are empty.
    """
    if not a and not b:
        return 0.0
    inter = len(a & b)
    return round(2 * inter / (len(a) + len(b)), 6)


def overlap_coefficient(a: set, b: set) -> float:
    """Return the Szymkiewicz-Simpson overlap coefficient between two sets.

    Args:
        a: First set.
        b: Second set.

    Returns:
        ``|a ∩ b| / min(|a|, |b|)``, or 0.0 when either set is empty.
    """
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return round(inter / min(len(a), len(b)), 6)


def pearson_correlation(a: list[float], b: list[float]) -> float:
    """Return the Pearson correlation coefficient between two series.

    Args:
        a: First series (length ≥ 2).
        b: Second series (same length as *a*).

    Returns:
        Coefficient in [-1, 1]; 0.0 when either series has zero variance.

    Raises:
        ValueError: If lengths differ or either series has fewer than 2 points.
    """
    if len(a) != len(b):
        raise ValueError(f"series must be same length, got {len(a)} and {len(b)}")
    if len(a) < 2:
        raise ValueError(f"series must have at least 2 points, got {len(a)}")
    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=False))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a == 0 or var_b == 0:
        return 0.0
    denom = (var_a * var_b) ** 0.5
    return round(num / denom, 6)


def hamming_distance(a: str, b: str) -> int:
    """Return the Hamming distance between two equal-length strings.

    Args:
        a: First string.
        b: Second string (must have the same length as *a*).

    Returns:
        Number of positions at which the strings differ.

    Raises:
        ValueError: If the strings have different lengths.
    """
    if len(a) != len(b):
        raise ValueError(f"strings must have the same length, got {len(a)} and {len(b)}")
    return sum(1 for ca, cb in zip(a, b, strict=False) if ca != cb)


def tanimoto_similarity(a: list[float], b: list[float]) -> float:
    """Return the Tanimoto similarity between two numeric vectors.

    Also known as the Extended Jaccard coefficient. Ranges from -1/3 to 1
    for arbitrary vectors, and from 0 to 1 for non-negative vectors.

    Args:
        a: First numeric vector.
        b: Second numeric vector (same length as *a*).

    Returns:
        Tanimoto coefficient in [-1/3, 1]; 0.0 when both vectors are all-zero.

    Raises:
        ValueError: If the vectors have different lengths.
    """
    if len(a) != len(b):
        raise ValueError(f"vectors must have same length, got {len(a)} and {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a)
    norm_b = sum(y * y for y in b)
    denom = norm_a + norm_b - dot
    if denom == 0:
        return 0.0
    return round(dot / denom, 6)


def vector_magnitude(a: list[float]) -> float:
    """Compute the L2 (Euclidean) norm (magnitude) of vector *a*.

    Args:
        a: Input vector.

    Returns:
        Non-negative magnitude; 0.0 for an empty or all-zero vector.
    """
    return round(sum(x * x for x in a) ** 0.5, 6)


def angular_distance(a: list[float], b: list[float]) -> float:
    """Compute angular distance (in radians) between two vectors.

    Angular distance is the arccosine of the cosine similarity, clipped to
    [0, pi] to handle floating-point rounding at the boundaries.

    Args:
        a: First vector.
        b: Second vector; must have the same length as *a*.

    Returns:
        Angular distance in [0.0, pi]; pi/2 when vectors are orthogonal.

    Raises:
        ValueError: If the vectors have different lengths.
    """
    import math

    if len(a) != len(b):
        raise ValueError(f"vectors must have same length, got {len(a)} and {len(b)}")
    mag_a = vector_magnitude(a)
    mag_b = vector_magnitude(b)
    if mag_a == 0.0 or mag_b == 0.0:
        return math.pi / 2.0
    dot = sum(x * y for x, y in zip(a, b))
    cos_sim = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
    return round(math.acos(cos_sim), 6)
