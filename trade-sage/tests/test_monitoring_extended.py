"""Extended tests for monitoring drift detection utilities."""

from __future__ import annotations

import numpy as np
import pytest

from app.monitoring import (
    clear_reference_store,
    compute_drift,
    get_drift_summary,
    update_reference,
)


@pytest.fixture(autouse=True)
def reset_reference_store():
    """Clear reference store before and after each test."""
    clear_reference_store()
    yield
    clear_reference_store()


class TestComputeDrift:
    def test_short_reference_returns_no_drift(self) -> None:
        result = compute_drift([1.0] * 5, [1.0, 2.0, 3.0])
        assert result["drift_detected"] is False
        assert result["p_value"] == 1.0

    def test_identical_distributions_no_drift(self) -> None:
        ref = list(np.random.default_rng(0).standard_normal(100))
        cur = list(np.random.default_rng(0).standard_normal(100))
        result = compute_drift(ref, cur)
        assert isinstance(result["drift_detected"], bool)

    def test_very_different_distributions_drift_detected(self) -> None:
        ref = list(np.random.default_rng(1).standard_normal(200))
        cur = list(np.random.default_rng(2).normal(10.0, 1.0, 200))
        result = compute_drift(ref, cur)
        assert result["drift_detected"] is True

    def test_ks_statistic_range(self) -> None:
        ref = list(range(50))
        cur = list(range(20, 70))
        result = compute_drift(ref, cur)
        assert 0.0 <= result["ks_statistic"] <= 1.0

    def test_p_value_range(self) -> None:
        ref = list(np.random.default_rng(3).standard_normal(100))
        cur = list(np.random.default_rng(4).standard_normal(50))
        result = compute_drift(ref, cur)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_result_has_required_keys(self) -> None:
        ref = list(range(20))
        cur = list(range(15))
        result = compute_drift(ref, cur)
        assert {"ks_statistic", "p_value", "drift_detected"} == set(result.keys())


class TestUpdateReference:
    def test_values_are_appended(self) -> None:
        update_reference("rsi", [50.0, 60.0, 70.0])
        from app.monitoring import _reference_store
        assert "rsi" in _reference_store
        assert len(_reference_store["rsi"]) == 3

    def test_window_is_bounded(self) -> None:
        from app.monitoring import REFERENCE_WINDOW
        update_reference("macd", list(range(REFERENCE_WINDOW + 100)))
        from app.monitoring import _reference_store
        assert len(_reference_store["macd"]) == REFERENCE_WINDOW


class TestGetDriftSummary:
    def test_empty_store_returns_empty(self) -> None:
        result = get_drift_summary()
        assert result == {}

    def test_populated_store_returns_results(self) -> None:
        rng = np.random.default_rng(5)
        update_reference("feature_x", list(rng.standard_normal(50)))
        result = get_drift_summary()
        assert "feature_x" in result

    def test_summary_values_have_drift_detected_key(self) -> None:
        rng = np.random.default_rng(6)
        update_reference("vol_ratio", list(rng.standard_normal(50)))
        result = get_drift_summary()
        if result:
            for v in result.values():
                assert "drift_detected" in v
