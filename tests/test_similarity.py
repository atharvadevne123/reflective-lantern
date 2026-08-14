"""Building similarity index tests."""

from __future__ import annotations

import pytest

from app.similarity import BuildingSimilarityIndex, chebyshev_distance, manhattan_distance, pearson_similarity


def test_empty_index_returns_empty() -> None:
    idx = BuildingSimilarityIndex()
    assert idx.search([1.0, 2.0, 3.0]) == []


def test_add_and_search() -> None:
    idx = BuildingSimilarityIndex()
    idx.add("A", [1.0, 0.0, 0.0])
    idx.add("B", [0.0, 1.0, 0.0])
    results = idx.search([1.0, 0.0, 0.0], k=1)
    assert results[0][0] == "A"
    assert results[0][1] > 0.9


def test_similarity_ordering() -> None:
    idx = BuildingSimilarityIndex()
    idx.add("close", [1.0, 1.0, 0.0])
    idx.add("far", [0.0, 0.0, 1.0])
    results = idx.search([1.0, 1.0, 0.0])
    assert results[0][0] == "close"


def test_len() -> None:
    idx = BuildingSimilarityIndex()
    idx.add("x", [1.0])
    idx.add("y", [2.0])
    assert len(idx) == 2


def test_k_limit() -> None:
    idx = BuildingSimilarityIndex()
    for i in range(10):
        idx.add(f"b{i}", [float(i), float(i + 1)])
    results = idx.search([5.0, 6.0], k=3)
    assert len(results) == 3


def test_clear_resets_index() -> None:
    idx = BuildingSimilarityIndex()
    idx.add("x", [1.0, 2.0])
    idx.add("y", [3.0, 4.0])
    idx.clear()
    assert len(idx) == 0
    assert idx.search([1.0, 2.0]) == []


def test_scores_between_neg1_and_1() -> None:
    idx = BuildingSimilarityIndex()
    idx.add("a", [1.0, 0.0])
    idx.add("b", [-1.0, 0.0])
    for _, score in idx.search([1.0, 0.0]):
        assert -1.0 <= score <= 1.0


def test_identical_vectors_score_one() -> None:
    idx = BuildingSimilarityIndex()
    idx.add("same", [3.0, 4.0])
    results = idx.search([3.0, 4.0], k=1)
    assert abs(results[0][1] - 1.0) < 1e-5


def test_k_larger_than_index() -> None:
    idx = BuildingSimilarityIndex()
    idx.add("only", [1.0])
    results = idx.search([1.0], k=100)
    assert len(results) == 1


def test_search_result_tuples_have_two_elements() -> None:
    idx = BuildingSimilarityIndex()
    idx.add("a", [1.0, 2.0])
    for item in idx.search([1.0, 2.0]):
        assert len(item) == 2


def test_euclidean_distance_same_vector() -> None:
    from app.similarity import euclidean_distance

    assert euclidean_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_euclidean_distance_different_vectors() -> None:
    from app.similarity import euclidean_distance

    d = euclidean_distance([0.0, 0.0], [3.0, 4.0])
    assert d == pytest.approx(5.0, rel=1e-4)


def test_euclidean_distance_negative_values() -> None:
    from app.similarity import euclidean_distance

    d = euclidean_distance([-1.0, -1.0], [1.0, 1.0])
    assert d > 0


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ([0.0], [0.0], 0.0),
        ([1.0, 0.0], [0.0, 1.0], pytest.approx(1.414, rel=1e-3)),
    ],
)
def test_euclidean_distance_parametrized(a, b, expected) -> None:
    from app.similarity import euclidean_distance

    assert euclidean_distance(a, b) == expected


def test_batch_add_all_successful() -> None:
    from app.similarity import BuildingSimilarityIndex, batch_add

    idx = BuildingSimilarityIndex()
    profiles = [("b1", [1.0, 0.0]), ("b2", [0.0, 1.0]), ("b3", [1.0, 1.0])]
    added = batch_add(idx, profiles)
    assert added == 3
    assert len(idx) == 3


def test_batch_add_empty() -> None:
    from app.similarity import BuildingSimilarityIndex, batch_add

    idx = BuildingSimilarityIndex()
    assert batch_add(idx, []) == 0


def test_score_distribution_empty_index() -> None:
    from app.similarity import BuildingSimilarityIndex, score_distribution

    idx = BuildingSimilarityIndex()
    result = score_distribution(idx, [1.0, 0.0, 0.0])
    assert result == {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}


def test_score_distribution_single_profile() -> None:
    from app.similarity import BuildingSimilarityIndex, score_distribution

    idx = BuildingSimilarityIndex()
    idx.add("b1", [1.0, 0.0])
    result = score_distribution(idx, [1.0, 0.0])
    assert result["max"] == pytest.approx(1.0, abs=1e-3)
    assert result["min"] == result["max"]


def test_score_distribution_keys() -> None:
    from app.similarity import BuildingSimilarityIndex, score_distribution

    idx = BuildingSimilarityIndex()
    idx.add("b1", [1.0, 0.0])
    idx.add("b2", [0.0, 1.0])
    result = score_distribution(idx, [1.0, 0.0])
    assert set(result.keys()) == {"min", "max", "mean", "std"}


def test_score_distribution_min_le_max() -> None:
    from app.similarity import BuildingSimilarityIndex, score_distribution

    idx = BuildingSimilarityIndex()
    for i in range(5):
        idx.add(f"b{i}", [float(i), float(i + 1), 0.5])
    result = score_distribution(idx, [1.0, 2.0, 0.5])
    assert result["min"] <= result["max"]


@pytest.mark.parametrize("n", [1, 5, 10, 50])
def test_score_distribution_various_sizes(n) -> None:
    import numpy as np

    from app.similarity import BuildingSimilarityIndex, score_distribution

    rng = np.random.default_rng(42)
    idx = BuildingSimilarityIndex()
    for i in range(n):
        idx.add(f"b{i}", rng.uniform(0, 1, 8).tolist())
    query = rng.uniform(0, 1, 8).tolist()
    result = score_distribution(idx, query)
    assert -1.0 <= result["min"] <= result["max"] <= 1.0


def test_hourly_pattern_distance_identical() -> None:
    from app.similarity import hourly_pattern_distance

    profile = [float(i) for i in range(24)]
    assert hourly_pattern_distance(profile, profile) == pytest.approx(0.0)


def test_hourly_pattern_distance_offset() -> None:
    from app.similarity import hourly_pattern_distance

    a = [0.0] * 24
    b = [1.0] * 24
    assert hourly_pattern_distance(a, b) == pytest.approx(1.0)


def test_hourly_pattern_distance_empty_raises() -> None:
    from app.similarity import hourly_pattern_distance

    with pytest.raises(ValueError):
        hourly_pattern_distance([], [])


def test_hourly_pattern_distance_mismatched_length_raises() -> None:
    from app.similarity import hourly_pattern_distance

    with pytest.raises(ValueError):
        hourly_pattern_distance([1.0] * 24, [1.0] * 12)


def test_hourly_pattern_distance_non_negative() -> None:
    from app.similarity import hourly_pattern_distance

    a = [float(i % 24) for i in range(24)]
    b = [float((i + 12) % 24) for i in range(24)]
    assert hourly_pattern_distance(a, b) >= 0.0


@pytest.mark.parametrize("offset", [0.5, 1.0, 2.0, 5.0])
def test_hourly_pattern_distance_constant_offset(offset) -> None:
    from app.similarity import hourly_pattern_distance

    a = [10.0] * 24
    b = [10.0 + offset] * 24
    assert hourly_pattern_distance(a, b) == pytest.approx(offset)


def test_building_similarity_index_add_and_size() -> None:
    from app.similarity import BuildingSimilarityIndex

    idx = BuildingSimilarityIndex()
    assert idx.size == 0
    idx.add("b1", [1.0] * 24)
    assert idx.size == 1


def test_building_similarity_index_clear() -> None:
    from app.similarity import BuildingSimilarityIndex

    idx = BuildingSimilarityIndex()
    idx.add("b1", [1.0] * 24)
    idx.clear()
    assert idx.size == 0


def test_building_similarity_search_returns_list() -> None:
    from app.similarity import BuildingSimilarityIndex

    idx = BuildingSimilarityIndex()
    for i in range(5):
        idx.add(f"b{i}", [float(i)] * 24)
    results = idx.search([2.0] * 24, k=3)
    assert isinstance(results, list)
    assert len(results) <= 3


def test_batch_add_returns_count() -> None:
    from app.similarity import BuildingSimilarityIndex, batch_add

    idx = BuildingSimilarityIndex()
    profiles = [(f"b{i}", [float(i)] * 24) for i in range(5)]
    count = batch_add(idx, profiles)
    assert count == 5


@pytest.mark.parametrize("k", [1, 3, 5])
def test_search_comparable_top_k(k: int) -> None:
    from app.similarity import get_global_index

    idx = get_global_index()
    idx.clear()
    for i in range(10):
        idx.add(f"b{i}", [float(i + 1)] * 24)
    from app.similarity import search_comparable

    results = search_comparable([5.0] * 24, top_k=k)
    assert len(results) <= k
    idx.clear()


def test_cosine_distance_identical_vectors() -> None:
    from app.similarity import cosine_distance

    assert cosine_distance([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-6)


def test_cosine_distance_orthogonal_vectors() -> None:
    from app.similarity import cosine_distance

    assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0, abs=1e-6)


def test_cosine_distance_range() -> None:
    from app.similarity import cosine_distance

    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    d = cosine_distance(a, b)
    assert 0.0 <= d <= 2.0


@pytest.mark.parametrize("n_dims", [3, 10, 24])
def test_cosine_distance_same_vector_various_dims(n_dims: int) -> None:
    from app.similarity import cosine_distance

    v = [1.0] * n_dims
    assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-6)


def test_building_similarity_index_search_returns_tuples() -> None:
    from app.similarity import BuildingSimilarityIndex

    idx = BuildingSimilarityIndex()
    idx.add("bld-A", [1.0, 2.0, 3.0])
    results = idx.search([1.0, 2.0, 3.0], k=1)
    assert len(results) == 1
    assert isinstance(results[0], tuple)
    assert results[0][0] == "bld-A"


class TestJaccardSimilarity:
    def test_identical_sets(self) -> None:
        from app.similarity import jaccard_similarity

        assert jaccard_similarity({"a", "b", "c"}, {"a", "b", "c"}) == pytest.approx(1.0, rel=1e-4)

    def test_disjoint_sets(self) -> None:
        from app.similarity import jaccard_similarity

        assert jaccard_similarity({"a", "b"}, {"c", "d"}) == pytest.approx(0.0, abs=1e-6)

    def test_partial_overlap(self) -> None:
        from app.similarity import jaccard_similarity

        result = jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        assert 0.0 < result < 1.0

    def test_both_empty(self) -> None:
        from app.similarity import jaccard_similarity

        assert jaccard_similarity(set(), set()) == 0.0

    def test_one_empty(self) -> None:
        from app.similarity import jaccard_similarity

        assert jaccard_similarity(set(), {"a", "b"}) == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("overlap", [1, 2, 3])
    def test_symmetric(self, overlap: int) -> None:
        from app.similarity import jaccard_similarity

        a = set("abc"[:overlap])
        b = set("bcd"[:overlap])
        assert jaccard_similarity(a, b) == jaccard_similarity(b, a)


class TestNormalizeDistances:
    def test_basic(self) -> None:
        from app.similarity import normalize_distances

        result = normalize_distances([0.0, 5.0, 10.0])
        assert result[0] == pytest.approx(0.0, abs=1e-6)
        assert result[-1] == pytest.approx(1.0, rel=1e-4)

    def test_empty_raises(self) -> None:
        from app.similarity import normalize_distances

        with pytest.raises(ValueError, match="empty"):
            normalize_distances([])

    def test_equal_values_returns_half(self) -> None:
        from app.similarity import normalize_distances

        result = normalize_distances([3.0, 3.0, 3.0])
        assert all(v == 0.5 for v in result)

    def test_output_length(self) -> None:
        from app.similarity import normalize_distances

        assert len(normalize_distances([1.0, 2.0, 3.0, 4.0])) == 4

    @pytest.mark.parametrize("distances", [[0.1, 0.5, 0.9], [10.0, 20.0, 30.0]])
    def test_values_in_range(self, distances: list) -> None:
        from app.similarity import normalize_distances

        result = normalize_distances(distances)
        assert all(0.0 <= v <= 1.0 for v in result)


class TestManhattanDistance:
    def test_basic(self) -> None:
        from app.similarity import manhattan_distance

        assert manhattan_distance([0.0, 0.0], [3.0, 4.0]) == pytest.approx(7.0, rel=1e-4)

    def test_identical_vectors(self) -> None:
        from app.similarity import manhattan_distance

        assert manhattan_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0, abs=1e-6)

    def test_empty_raises(self) -> None:
        from app.similarity import manhattan_distance

        with pytest.raises(ValueError, match="empty"):
            manhattan_distance([], [])

    def test_length_mismatch_raises(self) -> None:
        from app.similarity import manhattan_distance

        with pytest.raises(ValueError, match="same length"):
            manhattan_distance([1.0, 2.0], [1.0])

    @pytest.mark.parametrize("a,b,expected", [([1.0], [4.0], 3.0), ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 3.0)])
    def test_parametrized(self, a: list, b: list, expected: float) -> None:
        from app.similarity import manhattan_distance

        assert manhattan_distance(a, b) == pytest.approx(expected, rel=1e-4)


class TestTopKSimilar:
    def test_basic(self) -> None:
        from app.similarity import top_k_similar

        query = [1.0, 0.0, 0.0]
        candidates = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        result = top_k_similar(query, candidates, k=1)
        assert len(result) == 1
        assert result[0][0] == 0

    def test_empty_query_raises(self) -> None:
        from app.similarity import top_k_similar

        with pytest.raises(ValueError, match="empty"):
            top_k_similar([], [[1.0, 2.0]], k=1)

    def test_k_zero_raises(self) -> None:
        from app.similarity import top_k_similar

        with pytest.raises(ValueError, match="at least 1"):
            top_k_similar([1.0, 2.0], [[1.0, 2.0]], k=0)

    def test_result_sorted_ascending(self) -> None:
        from app.similarity import top_k_similar

        query = [1.0, 0.0]
        candidates = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
        result = top_k_similar(query, candidates, k=3)
        dists = [d for _, d in result]
        assert dists == sorted(dists)


class TestCosineDistance:
    def test_identical_vectors(self) -> None:
        from app.similarity import cosine_distance

        result = cosine_distance([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors(self) -> None:
        from app.similarity import cosine_distance

        result = cosine_distance([1.0, 0.0], [0.0, 1.0])
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_opposite_vectors(self) -> None:
        from app.similarity import cosine_distance

        result = cosine_distance([1.0, 0.0], [-1.0, 0.0])
        assert result == pytest.approx(2.0, rel=1e-4)

    def test_result_in_range(self) -> None:
        from app.similarity import cosine_distance

        result = cosine_distance([1.0, 0.0], [0.5, 0.5])
        assert 0.0 <= result <= 2.0

    def test_length_mismatch_raises(self) -> None:
        from app.similarity import cosine_distance

        with pytest.raises(ValueError):
            cosine_distance([1.0, 2.0], [1.0])


class TestWeightedJaccardSimilarity:
    def test_identical_dicts(self) -> None:
        from app.similarity import weighted_jaccard_similarity

        d = {"a": 1.0, "b": 2.0}
        assert weighted_jaccard_similarity(d, d) == pytest.approx(1.0)

    def test_disjoint_dicts(self) -> None:
        from app.similarity import weighted_jaccard_similarity

        a = {"x": 1.0}
        b = {"y": 1.0}
        assert weighted_jaccard_similarity(a, b) == pytest.approx(0.0)

    def test_empty_dicts(self) -> None:
        from app.similarity import weighted_jaccard_similarity

        assert weighted_jaccard_similarity({}, {}) == 0.0

    def test_partial_overlap(self) -> None:
        from app.similarity import weighted_jaccard_similarity

        a = {"a": 2.0, "b": 0.0}
        b = {"a": 1.0, "c": 1.0}
        result = weighted_jaccard_similarity(a, b)
        assert 0.0 <= result <= 1.0


def test_manhattan_distance_basic() -> None:
    assert manhattan_distance([0.0, 0.0], [3.0, 4.0]) == pytest.approx(7.0)


def test_manhattan_distance_identical() -> None:
    assert manhattan_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_manhattan_distance_negative() -> None:
    result = manhattan_distance([-1.0, -2.0], [1.0, 2.0])
    assert result == pytest.approx(6.0)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ([1.0, 0.0], [0.0, 1.0], 2.0),
        ([0.0, 0.0], [0.0, 0.0], 0.0),
    ],
)
def test_manhattan_parametrize(a, b, expected) -> None:
    assert manhattan_distance(a, b) == pytest.approx(expected)


def test_pearson_similarity_perfect_positive() -> None:
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = pearson_similarity(a, a)
    assert result == pytest.approx(1.0, abs=1e-4)


def test_pearson_similarity_perfect_negative() -> None:
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [5.0, 4.0, 3.0, 2.0, 1.0]
    result = pearson_similarity(a, b)
    assert result == pytest.approx(-1.0, abs=1e-4)


def test_pearson_similarity_constant_series() -> None:
    assert pearson_similarity([5.0] * 5, [3.0] * 5) == 0.0


def test_pearson_similarity_empty_raises() -> None:
    with pytest.raises(ValueError):
        pearson_similarity([], [])


def test_pearson_similarity_different_lengths_raises() -> None:
    with pytest.raises(ValueError):
        pearson_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


def test_chebyshev_distance_basic() -> None:
    result = chebyshev_distance([1.0, 2.0, 3.0], [4.0, 2.0, 1.0])
    assert result == pytest.approx(3.0)


def test_chebyshev_distance_identical() -> None:
    assert chebyshev_distance([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ([0.0, 0.0], [3.0, 4.0], 4.0),
        ([5.0, 5.0], [0.0, 0.0], 5.0),
    ],
)
def test_chebyshev_parametrize(a, b, expected) -> None:
    assert chebyshev_distance(a, b) == pytest.approx(expected)


class TestBuildingIds:
    def test_empty_index(self) -> None:
        idx = BuildingSimilarityIndex()
        assert idx.building_ids == []

    def test_preserves_insertion_order(self) -> None:
        idx = BuildingSimilarityIndex()
        idx.add("b1", [1.0, 0.0])
        idx.add("b2", [0.0, 1.0])
        idx.add("b3", [1.0, 1.0])
        assert idx.building_ids == ["b1", "b2", "b3"]

    def test_after_clear(self) -> None:
        idx = BuildingSimilarityIndex()
        idx.add("b1", [1.0, 0.0])
        idx.clear()
        assert idx.building_ids == []


class TestSimilarityMatrix:
    def test_diagonal_is_one(self) -> None:
        from app.similarity import similarity_matrix

        mat = similarity_matrix([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        for i in range(3):
            assert mat[i][i] == pytest.approx(1.0, abs=1e-4)

    def test_symmetric(self) -> None:
        from app.similarity import similarity_matrix

        mat = similarity_matrix([[1.0, 2.0], [3.0, 4.0]])
        assert mat[0][1] == pytest.approx(mat[1][0], abs=1e-6)

    def test_orthogonal_vectors_zero_similarity(self) -> None:
        from app.similarity import similarity_matrix

        mat = similarity_matrix([[1.0, 0.0], [0.0, 1.0]])
        assert mat[0][1] == pytest.approx(0.0, abs=1e-4)

    def test_empty_raises(self) -> None:
        from app.similarity import similarity_matrix

        with pytest.raises(ValueError):
            similarity_matrix([])


class TestMinkowskiDistance:
    def test_p2_equals_euclidean(self) -> None:
        from app.similarity import euclidean_distance, minkowski_distance

        a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
        assert minkowski_distance(a, b, p=2.0) == pytest.approx(euclidean_distance(a, b), abs=1e-5)

    def test_p1_equals_manhattan(self) -> None:
        from app.similarity import manhattan_distance, minkowski_distance

        a, b = [1.0, 2.0, 3.0], [4.0, 6.0, 9.0]
        assert minkowski_distance(a, b, p=1.0) == pytest.approx(manhattan_distance(a, b), abs=1e-5)

    def test_p_less_than_one_raises(self) -> None:
        from app.similarity import minkowski_distance

        with pytest.raises(ValueError):
            minkowski_distance([1.0], [2.0], p=0.5)

    def test_different_lengths_raises(self) -> None:
        from app.similarity import minkowski_distance

        with pytest.raises(ValueError):
            minkowski_distance([1.0, 2.0], [3.0])

    @pytest.mark.parametrize(
        "a,b,p,expected",
        [
            ([0.0, 0.0], [3.0, 4.0], 2.0, 5.0),
            ([0.0, 0.0], [3.0, 4.0], 1.0, 7.0),
        ],
    )
    def test_parametrized(self, a, b, p, expected) -> None:
        from app.similarity import minkowski_distance

        assert minkowski_distance(a, b, p) == pytest.approx(expected, abs=1e-4)


class TestDiceSimilarity:
    def test_identical_sets(self) -> None:
        from app.similarity import dice_similarity

        assert dice_similarity({"a", "b"}, {"a", "b"}) == pytest.approx(1.0)

    def test_disjoint_sets(self) -> None:
        from app.similarity import dice_similarity

        assert dice_similarity({"a"}, {"b"}) == pytest.approx(0.0)

    def test_empty_sets(self) -> None:
        from app.similarity import dice_similarity

        assert dice_similarity(set(), set()) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        from app.similarity import dice_similarity

        result = dice_similarity({"a", "b"}, {"b", "c"})
        assert 0.0 < result < 1.0


class TestOverlapCoefficient:
    def test_one_empty_returns_zero(self) -> None:
        from app.similarity import overlap_coefficient

        assert overlap_coefficient(set(), {"a"}) == pytest.approx(0.0)

    def test_subset_returns_one(self) -> None:
        from app.similarity import overlap_coefficient

        assert overlap_coefficient({"a"}, {"a", "b", "c"}) == pytest.approx(1.0)

    def test_disjoint_returns_zero(self) -> None:
        from app.similarity import overlap_coefficient

        assert overlap_coefficient({"a", "b"}, {"c", "d"}) == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            ({"x"}, {"x"}, 1.0),
            ({"x", "y"}, {"y", "z"}, 0.5),
        ],
    )
    def test_parametrized(self, a, b, expected) -> None:
        from app.similarity import overlap_coefficient

        assert overlap_coefficient(a, b) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Tests for euclidean_distance, pearson_correlation, top_k_similar
# ---------------------------------------------------------------------------


class TestEuclideanDistance:
    def test_zero_distance(self) -> None:
        from app.similarity import euclidean_distance

        assert euclidean_distance([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_known_distance(self) -> None:
        from app.similarity import euclidean_distance

        assert euclidean_distance([0.0, 0.0], [3.0, 4.0]) == pytest.approx(5.0, abs=0.001)

    def test_empty_raises(self) -> None:
        import pytest

        from app.similarity import euclidean_distance

        with pytest.raises(ValueError):
            euclidean_distance([], [])

    def test_length_mismatch_raises(self) -> None:
        import pytest

        from app.similarity import euclidean_distance

        with pytest.raises(ValueError):
            euclidean_distance([1.0], [1.0, 2.0])


class TestPearsonCorrelation:
    def test_perfect_positive(self) -> None:
        from app.similarity import pearson_correlation

        x = [1.0, 2.0, 3.0, 4.0]
        assert pearson_correlation(x, x) == pytest.approx(1.0, abs=0.001)

    def test_perfect_negative(self) -> None:
        from app.similarity import pearson_correlation

        x = [1.0, 2.0, 3.0]
        y = [3.0, 2.0, 1.0]
        assert pearson_correlation(x, y) == pytest.approx(-1.0, abs=0.001)

    def test_zero_variance_returns_zero(self) -> None:
        from app.similarity import pearson_correlation

        assert pearson_correlation([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]) == 0.0

    def test_too_short_raises(self) -> None:
        import pytest

        from app.similarity import pearson_correlation

        with pytest.raises(ValueError):
            pearson_correlation([1.0], [1.0])


class TestTopKSimilarExtended:
    def test_returns_k_results(self) -> None:
        from app.similarity import top_k_similar

        query = [1.0, 0.0]
        candidates = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        result = top_k_similar(query, candidates, k=2)
        assert len(result) == 2

    def test_most_similar_first(self) -> None:
        from app.similarity import top_k_similar

        query = [1.0, 0.0]
        candidates = [[1.0, 0.0], [0.0, 1.0]]
        result = top_k_similar(query, candidates, k=2)
        assert result[0][0] == 0  # first candidate is most similar

    def test_empty_candidates_raises(self) -> None:
        import pytest

        from app.similarity import top_k_similar

        with pytest.raises(ValueError):
            top_k_similar([1.0], [], k=1)

    def test_invalid_k_raises(self) -> None:
        import pytest

        from app.similarity import top_k_similar

        with pytest.raises(ValueError):
            top_k_similar([1.0], [[1.0]], k=0)
