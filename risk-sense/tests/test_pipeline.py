"""Integration tests for the full sklearn feature pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from app.features import (
    ClipOutlierTransformer,
    InteractionFeatureTransformer,
    LogIncomeTransformer,
    build_pipeline,
    extract_raw_features,
)


def _sample_X(n: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(
        [0, 0, 0, 1, 1, 0, 10_000],
        [1, 1, 1, 360, 30, 10, 500_000],
        size=(n, 7),
    )


class TestLogIncomeTransformer:
    def test_output_shape_unchanged(self) -> None:
        X = _sample_X(5)
        t = LogIncomeTransformer()
        out = t.fit_transform(X)
        assert out.shape == X.shape

    def test_income_column_changed(self) -> None:
        X = _sample_X(5)
        t = LogIncomeTransformer()
        out = t.fit_transform(X)
        assert not np.allclose(out[:, 6], X[:, 6])

    def test_other_columns_unchanged(self) -> None:
        X = _sample_X(5)
        t = LogIncomeTransformer()
        out = t.fit_transform(X)
        for col in range(6):
            np.testing.assert_array_equal(out[:, col], X[:, col])

    def test_zero_income_no_error(self) -> None:
        X = np.array([[0.3, 0.8, 0.4, 24, 5, 1, 0.0]])
        t = LogIncomeTransformer()
        out = t.fit_transform(X)
        assert out[0, 6] == pytest.approx(0.0)


class TestInteractionFeatureTransformer:
    def test_output_adds_two_columns(self) -> None:
        X = _sample_X(5)
        t = InteractionFeatureTransformer()
        out = t.fit_transform(X)
        assert out.shape == (5, 9)

    def test_interaction_column_is_product(self) -> None:
        X = np.array([[0.3, 0.8, 0.5, 24, 4, 2, 50000.0]])
        t = InteractionFeatureTransformer()
        out = t.fit_transform(X)
        expected_interaction = 0.5 * 2
        assert out[0, 7] == pytest.approx(expected_interaction)


class TestClipOutlierTransformer:
    def test_values_clipped_within_percentile_bounds(self) -> None:
        X = _sample_X(100)
        t = ClipOutlierTransformer(lower_pct=5.0, upper_pct=95.0)
        t.fit(X)
        out = t.transform(X)
        assert out.min() >= t.lower_.min() - 1e-9
        assert out.max() <= t.upper_.max() + 1e-9

    def test_transform_before_fit_raises(self) -> None:
        X = _sample_X(5)
        t = ClipOutlierTransformer()
        with pytest.raises(Exception):
            t.transform(X)


class TestBuildPipeline:
    def test_pipeline_fit_transform(self) -> None:
        X = _sample_X(50)
        pipeline = build_pipeline()
        out = pipeline.fit_transform(X)
        assert out.shape[0] == 50
        assert out.shape[1] == 9

    def test_pipeline_has_four_steps(self) -> None:
        pipeline = build_pipeline()
        assert len(pipeline.steps) == 4

    def test_pipeline_produces_finite_values(self) -> None:
        X = _sample_X(20)
        pipeline = build_pipeline()
        out = pipeline.fit_transform(X)
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("n_samples", [1, 5, 100])
    def test_various_batch_sizes(self, n_samples: int) -> None:
        X_train = _sample_X(100)
        X_test = _sample_X(n_samples, seed=99)
        pipeline = build_pipeline()
        pipeline.fit(X_train)
        out = pipeline.transform(X_test)
        assert out.shape[0] == n_samples
