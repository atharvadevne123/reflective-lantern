"""Tests for the explainability module."""

from __future__ import annotations

import numpy as np
import pytest

from app.explainability import (
    adverse_action_reasons,
    score_band_description,
    top_risk_drivers,
)


class TestTopRiskDrivers:
    def test_returns_top_n_entries(self) -> None:
        values = np.array([0.5, 0.8, 0.3, 24.0, 5.0, 2.0, 10.5, 0.6, 2.1])
        importances = np.array([0.2, 0.15, 0.25, 0.05, 0.05, 0.1, 0.08, 0.07, 0.05])
        result = top_risk_drivers(values, importances, top_n=3)
        assert len(result) == 3

    def test_result_has_expected_keys(self) -> None:
        values = np.array([0.5, 0.8, 0.3, 24.0, 5.0, 2.0, 10.5, 0.6, 2.1])
        importances = np.ones(9) / 9
        result = top_risk_drivers(values, importances, top_n=2)
        for entry in result:
            assert "feature" in entry
            assert "contribution" in entry
            assert "explanation" in entry

    def test_2d_input_handled(self) -> None:
        values = np.array([[0.5, 0.8, 0.3, 24.0, 5.0, 2.0, 10.5, 0.6, 2.1]])
        importances = np.ones(9) / 9
        result = top_risk_drivers(values, importances, top_n=2)
        assert len(result) == 2

    def test_mismatched_lengths_raise(self) -> None:
        values = np.array([0.1, 0.2])
        importances = np.array([0.5, 0.3, 0.2])
        with pytest.raises(ValueError):
            top_risk_drivers(values, importances)

    @pytest.mark.parametrize("top_n", [1, 3, 5])
    def test_various_top_n(self, top_n: int) -> None:
        values = np.ones(9)
        importances = np.ones(9) / 9
        result = top_risk_drivers(values, importances, top_n=top_n)
        assert len(result) == top_n


class TestAdverseActionReasons:
    def test_no_reasons_when_below_threshold(self) -> None:
        values = np.ones(9) * 0.1
        importances = np.ones(9) / 9
        reasons = adverse_action_reasons(values, importances, risk_score=0.3)
        assert reasons == []

    def test_returns_reasons_for_high_risk(self) -> None:
        values = np.array([0.9, 0.1, 0.9, 10.0, 2.0, 5.0, 8.0, 0.9, 1.5])
        importances = np.array([0.2, 0.15, 0.25, 0.05, 0.05, 0.1, 0.08, 0.07, 0.05])
        reasons = adverse_action_reasons(values, importances, risk_score=0.8)
        assert isinstance(reasons, list)
        assert len(reasons) > 0
        assert all(isinstance(r, str) for r in reasons)

    def test_max_four_reasons_returned(self) -> None:
        values = np.ones(9)
        importances = np.ones(9) / 9
        reasons = adverse_action_reasons(values, importances, risk_score=0.9)
        assert len(reasons) <= 4


class TestScoreBandDescription:
    @pytest.mark.parametrize("score,expected_word", [
        (0.1, "Low"),
        (0.35, "Medium"),
        (0.6, "High"),
        (0.9, "Very high"),
    ])
    def test_score_band_labels(self, score: float, expected_word: str) -> None:
        desc = score_band_description(score)
        assert desc.startswith(expected_word)
