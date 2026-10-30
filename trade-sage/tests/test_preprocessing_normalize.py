"""Tests for normalize_features in preprocessing module."""

from __future__ import annotations

import numpy as np
import pytest

from app.preprocessing import normalize_features


class TestNormalizeFeatures:
    def test_output_range_zero_one(self) -> None:
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        result = normalize_features(X)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_shape_preserved(self) -> None:
        X = np.random.default_rng(0).random((20, 5))
        result = normalize_features(X)
        assert result.shape == X.shape

    def test_constant_column_set_to_zero(self) -> None:
        X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        result = normalize_features(X)
        assert np.all(result[:, 0] == 0.0)

    def test_single_row(self) -> None:
        X = np.array([[1.0, 2.0, 3.0]])
        result = normalize_features(X)
        assert result.shape == (1, 3)

    def test_returns_float64(self) -> None:
        X = np.array([[1, 2], [3, 4]], dtype=int)
        result = normalize_features(X)
        assert result.dtype == np.float64

    def test_first_col_min_max(self) -> None:
        X = np.array([[0.0, 100.0], [50.0, 50.0], [100.0, 0.0]])
        result = normalize_features(X)
        assert result[0, 0] == pytest.approx(0.0)
        assert result[2, 0] == pytest.approx(1.0)
