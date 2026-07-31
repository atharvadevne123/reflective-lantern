"""Building similarity index tests."""

from __future__ import annotations

import pytest

from app.similarity import BuildingSimilarityIndex


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
