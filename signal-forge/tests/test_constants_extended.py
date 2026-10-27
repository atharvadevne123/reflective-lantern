"""Extended tests for signal-forge app/constants.py."""

from __future__ import annotations

import pytest

from app.constants import (
    ANNUALISATION_FACTOR,
    BETA_ROLLING_WINDOW,
    DEFAULT_ROLLING_WINDOW,
    FAISS_DEFAULT_K,
    FEATURE_NAMES,
    MIN_DRIFT_SAMPLES,
    MODEL_VERSION,
    REGIME_RISK_WEIGHTS,
    REGIMES,
    RISK_TIERS,
)


class TestRegimes:
    def test_regimes_is_list(self) -> None:
        assert isinstance(REGIMES, list)

    def test_regimes_not_empty(self) -> None:
        assert len(REGIMES) > 0

    def test_known_regimes(self) -> None:
        assert "bull" in REGIMES
        assert "bear" in REGIMES

    def test_all_regimes_have_risk_weights(self) -> None:
        for regime in REGIMES:
            assert regime in REGIME_RISK_WEIGHTS

    def test_risk_weights_in_range(self) -> None:
        for weight in REGIME_RISK_WEIGHTS.values():
            assert 0.0 <= weight <= 1.0


class TestRiskTiers:
    def test_risk_tiers_is_dict(self) -> None:
        assert isinstance(RISK_TIERS, dict)

    def test_has_low_tier(self) -> None:
        assert "low" in RISK_TIERS

    def test_has_critical_tier(self) -> None:
        assert "critical" in RISK_TIERS

    def test_tier_ranges_are_tuples(self) -> None:
        for tier, bounds in RISK_TIERS.items():
            assert isinstance(bounds, tuple)
            assert len(bounds) == 2

    def test_tier_lower_lt_upper(self) -> None:
        for tier, (lo, hi) in RISK_TIERS.items():
            assert lo < hi


class TestFeatureNames:
    def test_feature_names_not_empty(self) -> None:
        assert len(FEATURE_NAMES) > 0

    def test_all_strings(self) -> None:
        for f in FEATURE_NAMES:
            assert isinstance(f, str)


class TestNumericalConstants:
    def test_annualisation_factor_positive(self) -> None:
        assert ANNUALISATION_FACTOR > 0.0

    def test_default_rolling_window_positive(self) -> None:
        assert DEFAULT_ROLLING_WINDOW > 0

    def test_beta_rolling_window_gt_default(self) -> None:
        assert BETA_ROLLING_WINDOW > DEFAULT_ROLLING_WINDOW

    def test_min_drift_samples_positive(self) -> None:
        assert MIN_DRIFT_SAMPLES > 0

    def test_faiss_default_k_positive(self) -> None:
        assert FAISS_DEFAULT_K > 0

    def test_model_version_string(self) -> None:
        assert isinstance(MODEL_VERSION, str)
        assert "." in MODEL_VERSION
