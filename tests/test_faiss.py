"""Tests for FAISS comparable property search."""

import numpy as np
import pytest

from app.faiss_index import (
    DIM,
    add_property,
    index_size,
    reset_index,
    search_comparable,
)


@pytest.fixture(autouse=True)
def fresh_index() -> None:
    reset_index()
    yield
    reset_index()


def make_vec(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, 10, DIM).astype(np.float32)


def test_index_starts_empty() -> None:
    assert index_size() == 0


def test_add_property_increases_size() -> None:
    add_property(make_vec(1), {"zipcode": "94102", "predicted_value": 500_000})
    assert index_size() == 1


def test_search_empty_returns_empty() -> None:
    results = search_comparable(make_vec(0), top_k=3)
    assert results == []


def test_search_returns_results() -> None:
    for i in range(5):
        add_property(make_vec(i), {"id": i, "predicted_value": i * 100_000})
    results = search_comparable(make_vec(0), top_k=3)
    assert len(results) == 3
    assert all("distance" in r for r in results)


def test_search_top_k_respected() -> None:
    for i in range(10):
        add_property(make_vec(i), {"id": i})
    results = search_comparable(make_vec(5), top_k=2)
    assert len(results) == 2


def test_search_top_k_capped_at_index_size() -> None:
    add_property(make_vec(0), {"id": 0})
    results = search_comparable(make_vec(0), top_k=10)
    assert len(results) == 1


def test_closest_vector_has_smallest_distance() -> None:
    query = make_vec(99)
    add_property(query.copy(), {"id": "exact"})
    add_property(make_vec(42), {"id": "far"})
    results = search_comparable(query, top_k=2)
    assert results[0]["distance"] <= results[1]["distance"]


def test_short_vector_is_padded() -> None:
    short = np.ones(5, dtype=np.float32)
    add_property(short, {"id": "short"})
    assert index_size() == 1


def test_long_vector_is_truncated() -> None:
    long_v = np.ones(50, dtype=np.float32)
    add_property(long_v, {"id": "long"})
    assert index_size() == 1


@pytest.mark.parametrize("n", [1, 3, 7])
def test_index_size_after_n_additions(n) -> None:
    for i in range(n):
        add_property(make_vec(i), {"id": i})
    assert index_size() == n


def test_reset_clears_all() -> None:
    for i in range(5):
        add_property(make_vec(i), {"id": i})
    reset_index()
    assert index_size() == 0
    assert search_comparable(make_vec(0)) == []


@pytest.mark.parametrize("top_k", [1, 3, 5])
def test_search_top_k_various(top_k: int) -> None:
    for i in range(10):
        add_property(make_vec(i), {"id": i})
    results = search_comparable(make_vec(0), top_k=top_k)
    assert len(results) == top_k


def test_search_results_have_metadata() -> None:
    add_property(make_vec(0), {"zipcode": "94102", "bedrooms": 3, "sqft": 1800})
    results = search_comparable(make_vec(0), top_k=1)
    assert len(results) == 1
    assert results[0]["zipcode"] == "94102"


def test_search_returns_sorted_by_distance() -> None:
    query = make_vec(99)
    add_property(query.copy(), {"id": "near"})
    far_vec = make_vec(99) + 100.0
    add_property(far_vec, {"id": "far"})
    results = search_comparable(query, top_k=2)
    assert results[0]["distance"] <= results[1]["distance"]


def test_add_then_reset_then_add_works() -> None:
    add_property(make_vec(0), {"id": 0})
    assert index_size() == 1
    reset_index()
    assert index_size() == 0
    add_property(make_vec(1), {"id": 1})
    assert index_size() == 1


@pytest.mark.parametrize("n", [1, 5, 20])
def test_search_top_k_capped_when_fewer_stored(n: int) -> None:
    for i in range(n):
        add_property(make_vec(i), {"id": i})
    results = search_comparable(make_vec(0), top_k=100)
    assert len(results) == n


def test_zero_vector_can_be_added() -> None:
    zero_vec = np.zeros(DIM, dtype=np.float32)
    add_property(zero_vec, {"id": "zero"})
    assert index_size() == 1
    results = search_comparable(zero_vec, top_k=1)
    assert len(results) == 1
