"""Tests for app/data/preprocessor.py."""

from __future__ import annotations

import pytest


def test_normalize_basic():
    from app.data.preprocessor import normalize

    result = normalize([0.0, 5.0, 10.0])
    assert result[0] == 0.0
    assert result[-1] == 1.0
    assert abs(result[1] - 0.5) < 1e-9


def test_normalize_empty():
    from app.data.preprocessor import normalize

    assert normalize([]) == []


def test_normalize_constant():
    from app.data.preprocessor import normalize

    result = normalize([3.0, 3.0, 3.0])
    assert all(v == 0.5 for v in result)


def test_standardize_basic():
    from app.data.preprocessor import standardize

    result = standardize([1.0, 2.0, 3.0])
    assert abs(sum(result)) < 1e-9  # mean ~ 0


def test_standardize_constant():
    from app.data.preprocessor import standardize

    result = standardize([4.0, 4.0, 4.0])
    assert all(v == 0.0 for v in result)


def test_encode_category():
    from app.data.preprocessor import encode_category

    result = encode_category(["b", "a", "c", "a"])
    assert result[1] == result[3]  # both 'a'
    assert len(set(result)) == 3


def test_fill_missing():
    from app.data.preprocessor import fill_missing

    result = fill_missing([1.0, None, 3.0], fill_value=-1.0)
    assert result[1] == -1.0
    assert result[0] == 1.0


def test_clip_outliers():
    from app.data.preprocessor import clip_outliers

    values = [1.0] * 9 + [1000.0]
    result = clip_outliers(values, z_threshold=2.0)
    assert result[-1] < 1000.0


@pytest.mark.parametrize("n", [5, 10, 50])
def test_normalize_length_preserved(n: int) -> None:
    from app.data.preprocessor import normalize

    values = [float(i) for i in range(n)]
    result = normalize(values)
    assert len(result) == n


@pytest.mark.parametrize("n", [5, 10, 50])
def test_standardize_length_preserved(n: int) -> None:
    from app.data.preprocessor import standardize

    values = [float(i) for i in range(n)]
    result = standardize(values)
    assert len(result) == n


@pytest.mark.parametrize(
    "categories,expected_count",
    [
        (["A", "B", "C"], 3),
        (["X", "X", "Y"], 3),
    ],
)
def test_encode_category_length(categories: list, expected_count: int) -> None:
    from app.data.preprocessor import encode_category

    result = encode_category(categories)
    assert len(result) == expected_count


def test_fill_missing_replaces_none() -> None:
    from app.data.preprocessor import fill_missing

    result = fill_missing([1.0, None, 3.0], fill_value=-1.0)
    assert result == [1.0, -1.0, 3.0]


def test_clip_outliers_empty_returns_empty() -> None:
    from app.data.preprocessor import clip_outliers

    assert clip_outliers([]) == []


def test_encode_category_empty_returns_empty() -> None:
    from app.data.preprocessor import encode_category

    assert encode_category([]) == []


@pytest.mark.parametrize(
    "values,expected",
    [
        ([1.0, 2.0, 3.0], [0.0, 0.5, 1.0]),
        ([5.0, 5.0], [0.5, 0.5]),
    ],
)
def test_normalize_range(values: list, expected: list) -> None:
    from app.data.preprocessor import normalize

    result = normalize(values)
    for r, e in zip(result, expected, strict=False):
        assert abs(r - e) < 1e-6


def test_standardize_mean_approx_zero() -> None:
    from app.data.preprocessor import standardize

    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = standardize(values)
    mean = sum(result) / len(result)
    assert abs(mean) < 1e-6


def test_standardize_empty_returns_empty() -> None:
    from app.data.preprocessor import standardize

    assert standardize([]) == []


@pytest.mark.parametrize("z_threshold", [2.0, 3.0, 4.0])
def test_clip_outliers_result_within_bounds(z_threshold: float) -> None:
    from app.data.preprocessor import clip_outliers

    values = [1.0] * 10 + [1000.0]
    result = clip_outliers(values, z_threshold=z_threshold)
    mean = sum(values) / len(values)
    import math

    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(variance)
    for v in result:
        assert v <= mean + z_threshold * std + 1e-6
