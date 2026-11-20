"""Extended tests for feature engineering including validate_feature_count."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import FEATURE_NAMES, validate_feature_count


def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 + rng.standard_normal(n).cumsum()
    close = np.abs(close) + 10
    return pd.DataFrame({
        "open": close * rng.uniform(0.99, 1.01, n),
        "high": close * rng.uniform(1.0, 1.02, n),
        "low": close * rng.uniform(0.98, 1.0, n),
        "close": close,
        "volume": rng.integers(100_000, 1_000_000, n).astype(float),
    })


class TestValidateFeatureCount:
    def test_correct_column_count_returns_true(self) -> None:
        X = np.zeros((10, len(FEATURE_NAMES)))
        assert validate_feature_count(X) is True

    def test_wrong_column_count_returns_false(self) -> None:
        X = np.zeros((10, 5))
        assert validate_feature_count(X) is False

    def test_one_d_array_returns_false(self) -> None:
        X = np.zeros(len(FEATURE_NAMES))
        assert validate_feature_count(X) is False

    def test_zero_columns_returns_false(self) -> None:
        X = np.zeros((10, 0))
        assert validate_feature_count(X) is False

    def test_extra_column_returns_false(self) -> None:
        X = np.zeros((10, len(FEATURE_NAMES) + 1))
        assert validate_feature_count(X) is False

    @pytest.mark.parametrize("n_cols", [1, 10, 21, 22, 23])
    def test_parametrized_column_counts(self, n_cols: int) -> None:
        X = np.zeros((5, n_cols))
        expected = n_cols == len(FEATURE_NAMES)
        assert validate_feature_count(X) is expected


class TestFeatureNames:
    def test_feature_names_is_list(self) -> None:
        assert isinstance(FEATURE_NAMES, list)

    def test_feature_names_length(self) -> None:
        assert len(FEATURE_NAMES) == 22

    def test_feature_names_are_strings(self) -> None:
        for name in FEATURE_NAMES:
            assert isinstance(name, str)

    def test_feature_names_unique(self) -> None:
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))

    def test_rsi_in_feature_names(self) -> None:
        assert "rsi" in FEATURE_NAMES

    def test_macd_in_feature_names(self) -> None:
        assert "macd" in FEATURE_NAMES
