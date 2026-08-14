"""Tests for app/data/preprocessor.py."""

from __future__ import annotations


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
