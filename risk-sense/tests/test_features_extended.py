"""Extended feature engineering tests for risk-sense with parametrize and boundary coverage."""

from __future__ import annotations

import numpy as np
import pytest

from app.features import (
    ClipOutlierTransformer,
    InteractionFeatureTransformer,
    LogIncomeTransformer,
)


def _base_matrix(n: int = 20, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.uniform(0, 5, n),  # 0: debt_to_income
            rng.uniform(0, 1, n),  # 1: payment_history_score
            rng.uniform(0, 1, n),  # 2: credit_utilization
            rng.integers(0, 300, n).astype(float),  # 3: account_age_months
            rng.integers(1, 20, n).astype(float),  # 4: num_open_accounts
            rng.integers(0, 5, n).astype(float),  # 5: num_derogatory_marks
            rng.exponential(50_000, n),  # 6: annual_income
        ]
    )


@pytest.mark.parametrize("n", [1, 10, 100])
def test_log_income_transformer_row_count(n: int) -> None:
    X = _base_matrix(n)
    out = LogIncomeTransformer().fit_transform(X)
    assert out.shape[0] == n


def test_log_income_transformer_col_count_unchanged() -> None:
    X = _base_matrix(10)
    out = LogIncomeTransformer().fit_transform(X)
    assert out.shape[1] == X.shape[1]


@pytest.mark.parametrize("income", [0.0, 1.0, 50_000.0, 100_000.0, 1_000_000.0])
def test_log_income_is_log1p(income: float) -> None:
    X = _base_matrix(1)
    X[0, 6] = income
    out = LogIncomeTransformer().fit_transform(X)
    assert out[0, 6] == pytest.approx(np.log1p(income), rel=1e-6)


def test_log_income_clips_negative_income() -> None:
    X = _base_matrix(1)
    X[0, 6] = -50_000.0
    out = LogIncomeTransformer().fit_transform(X)
    assert out[0, 6] == pytest.approx(np.log1p(0.0), abs=1e-9)


def test_log_income_other_cols_unchanged() -> None:
    X = _base_matrix(10)
    out = LogIncomeTransformer().fit_transform(X)
    for col in range(6):
        np.testing.assert_array_equal(out[:, col], X[:, col])


@pytest.mark.parametrize("n", [1, 10, 50])
def test_interaction_transformer_adds_two_cols(n: int) -> None:
    X = _base_matrix(n)
    out = InteractionFeatureTransformer().fit_transform(X)
    assert out.shape[1] == X.shape[1] + 2


def test_interaction_col_utilization_x_derogatory() -> None:
    X = _base_matrix(5)
    out = InteractionFeatureTransformer().fit_transform(X)
    expected = X[:, 2] * X[:, 5]
    np.testing.assert_allclose(out[:, -2], expected, rtol=1e-5)


def test_interaction_col_income_per_account() -> None:
    X = _base_matrix(5)
    out = InteractionFeatureTransformer().fit_transform(X)
    accounts = np.clip(X[:, 4], 1, None)
    expected = X[:, 6] / accounts
    np.testing.assert_allclose(out[:, -1], expected, rtol=1e-5)


def test_interaction_zero_accounts_clipped_to_one() -> None:
    X = _base_matrix(1)
    X[0, 4] = 0.0
    out = InteractionFeatureTransformer().fit_transform(X)
    assert np.isfinite(out[0, -1])


@pytest.mark.parametrize(
    "lower,upper",
    [
        (1.0, 99.0),
        (5.0, 95.0),
        (10.0, 90.0),
    ],
)
def test_clip_outlier_transformer_clipping_bounds(lower: float, upper: float) -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(0, 10, (200, 3))
    transformer = ClipOutlierTransformer(lower_pct=lower, upper_pct=upper)
    transformer.fit(X)
    out = transformer.transform(X)
    assert out.min() >= transformer.lower_.min() - 1e-6
    assert out.max() <= transformer.upper_.max() + 1e-6


def test_clip_outlier_transformer_not_fitted_raises() -> None:
    t = ClipOutlierTransformer()
    with pytest.raises(Exception):
        t.transform(np.array([[1.0, 2.0]]))


def test_clip_outlier_transformer_fit_sets_bounds() -> None:
    X = _base_matrix(100)
    t = ClipOutlierTransformer()
    t.fit(X)
    assert t.lower_ is not None
    assert t.upper_ is not None
    assert len(t.lower_) == X.shape[1]


def test_clip_outlier_no_nans_after_transform() -> None:
    X = _base_matrix(100)
    t = ClipOutlierTransformer()
    t.fit(X)
    out = t.transform(X)
    assert not np.isnan(out).any()
