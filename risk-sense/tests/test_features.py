"""Feature engineering pipeline tests for Risk-Sense."""

from __future__ import annotations

import numpy as np
import pytest

from app.features import (
    ClipOutlierTransformer,
    InteractionFeatureTransformer,
    LogIncomeTransformer,
    build_pipeline,
    extract_raw_features,
    feature_importance_report,
    risk_category,
)


class TestLogIncomeTransformer:
    def test_transforms_column_6(self) -> None:
        X = np.array([[0.3, 0.8, 0.2, 48, 4, 0, 50_000.0]])
        t = LogIncomeTransformer()
        result = t.fit_transform(X)
        assert result[0, 6] == pytest.approx(np.log1p(50_000.0))

    def test_other_columns_unchanged(self) -> None:
        X = np.array([[0.3, 0.8, 0.2, 48, 4, 0, 50_000.0]])
        result = LogIncomeTransformer().fit_transform(X)
        np.testing.assert_array_equal(result[0, :6], X[0, :6])

    def test_zero_income_no_error(self) -> None:
        X = np.array([[0.3, 0.8, 0.2, 48, 4, 0, 0.0]])
        result = LogIncomeTransformer().fit_transform(X)
        assert result[0, 6] == pytest.approx(0.0)

    def test_negative_income_clipped_to_zero(self) -> None:
        X = np.array([[0.3, 0.8, 0.2, 48, 4, 0, -100.0]])
        result = LogIncomeTransformer().fit_transform(X)
        assert result[0, 6] == pytest.approx(0.0)


class TestInteractionFeatureTransformer:
    def test_adds_two_columns(self) -> None:
        X = np.array([[0.3, 0.8, 0.2, 48, 4, 1, 1.0]])
        result = InteractionFeatureTransformer().fit_transform(X)
        assert result.shape[1] == X.shape[1] + 2

    def test_interaction_term_is_utilization_times_derogatory(self) -> None:
        X = np.array([[0.3, 0.8, 0.5, 48, 4, 2, 1.0]])
        result = InteractionFeatureTransformer().fit_transform(X)
        assert result[0, -2] == pytest.approx(0.5 * 2)

    def test_income_per_account_uses_log_income(self) -> None:
        X = np.array([[0.3, 0.8, 0.2, 48, 4, 0, 1.0]])
        result = InteractionFeatureTransformer().fit_transform(X)
        expected = X[0, 6] / 4  # income_log / num_accounts
        assert result[0, -1] == pytest.approx(expected)

    def test_zero_accounts_clipped_to_one(self) -> None:
        X = np.array([[0.3, 0.8, 0.2, 48, 0, 0, 10.0]])
        result = InteractionFeatureTransformer().fit_transform(X)
        assert np.isfinite(result[0, -1])


class TestClipOutlierTransformer:
    def test_fit_transform_clips_values(self) -> None:
        X = np.array([[i * 1.0] for i in range(100)])
        t = ClipOutlierTransformer(lower_pct=5.0, upper_pct=95.0)
        result = t.fit_transform(X)
        assert result.min() >= X.min() * 0.9

    def test_transform_uses_fitted_bounds(self) -> None:
        X_train = np.array([[i * 1.0] for i in range(100)])
        t = ClipOutlierTransformer(lower_pct=5.0, upper_pct=95.0)
        t.fit(X_train)
        X_test = np.array([[1000.0]])
        result = t.transform(X_test)
        assert result[0, 0] <= t.upper_[0] + 1e-9


class TestBuildPipeline:
    def test_pipeline_has_four_steps(self) -> None:
        p = build_pipeline()
        assert len(p.steps) == 4

    def test_pipeline_fit_transform_returns_array(self) -> None:
        X = np.array([[0.3, 0.8, 0.2, 48, 4, 0, 50_000.0]] * 20)
        p = build_pipeline()
        result = p.fit_transform(X)
        assert result.shape[0] == 20


class TestExtractRawFeatures:
    def test_extracts_7_features(self) -> None:
        app = {
            "debt_to_income": 0.3,
            "payment_history_score": 0.8,
            "credit_utilization": 0.2,
            "account_age_months": 48,
            "num_open_accounts": 4,
            "num_derogatory_marks": 0,
            "annual_income": 50_000.0,
        }
        X = extract_raw_features(app)
        assert X.shape == (1, 7)

    def test_missing_field_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Missing required fields"):
            extract_raw_features({"debt_to_income": 0.3})

    @pytest.mark.parametrize(
        "field,value",
        [
            ("debt_to_income", "0.5"),
            ("annual_income", "80000"),
        ],
    )
    def test_string_values_coerced_to_float(self, field: str, value: str) -> None:
        app = {
            "debt_to_income": "0.3",
            "payment_history_score": "0.8",
            "credit_utilization": "0.2",
            "account_age_months": "48",
            "num_open_accounts": "4",
            "num_derogatory_marks": "0",
            "annual_income": "50000.0",
        }
        app[field] = value
        X = extract_raw_features(app)
        assert X.shape == (1, 7)


class TestRiskCategory:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (0.0, "low"),
            (0.19, "low"),
            (0.2, "medium"),
            (0.49, "medium"),
            (0.5, "high"),
            (0.74, "high"),
            (0.75, "very_high"),
            (1.0, "very_high"),
        ],
    )
    def test_category_boundaries(self, score: float, expected: str) -> None:
        assert risk_category(score) == expected


class TestFeatureImportanceReport:
    def test_sorted_descending(self) -> None:
        result = feature_importance_report(["a", "b", "c"], [0.1, 0.5, 0.3])
        assert result[0]["feature"] == "b"
        assert result[0]["importance"] >= result[1]["importance"]

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError):
            feature_importance_report(["a", "b"], [0.5])

    def test_returns_list_of_dicts(self) -> None:
        result = feature_importance_report(["x"], [0.9])
        assert isinstance(result, list)
        assert "feature" in result[0]
        assert "importance" in result[0]
