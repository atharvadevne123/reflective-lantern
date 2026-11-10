"""Extended tests for suite-cast app/model.py predict_demand and generate_sample_data."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.model import generate_sample_data, predict_demand


class TestGenerateSampleData:
    def test_returns_tuple(self) -> None:
        X, y = generate_sample_data(100)
        assert X is not None
        assert y is not None

    def test_x_shape(self) -> None:
        X, y = generate_sample_data(50)
        assert len(X) == 50

    def test_y_shape(self) -> None:
        X, y = generate_sample_data(50)
        assert len(y) == 50

    def test_y_binary(self) -> None:
        X, y = generate_sample_data(200)
        unique = set(np.unique(y))
        assert unique.issubset({0, 1})


class TestPredictDemand:
    def _mock_model(self, probs: list[float]) -> MagicMock:
        m = MagicMock()
        m.predict_proba.return_value = np.array([[1 - p, p] for p in probs])
        return m

    def _make_X(self, n: int = 1):
        import pandas as pd
        return pd.DataFrame({"feature": [0.5] * n})

    def test_returns_dict(self) -> None:
        xgb = self._mock_model([0.7])
        result = predict_demand(xgb, None, self._make_X())
        assert isinstance(result, dict)

    def test_has_demand_score(self) -> None:
        xgb = self._mock_model([0.6])
        result = predict_demand(xgb, None, self._make_X())
        assert "demand_score" in result

    def test_has_demand_tier(self) -> None:
        xgb = self._mock_model([0.5])
        result = predict_demand(xgb, None, self._make_X())
        assert result["demand_tier"] in ("low", "medium", "high")

    def test_high_demand_tier(self) -> None:
        xgb = self._mock_model([0.95])
        result = predict_demand(xgb, None, self._make_X())
        assert result["demand_tier"] == "high"

    def test_low_demand_tier(self) -> None:
        xgb = self._mock_model([0.05])
        result = predict_demand(xgb, None, self._make_X())
        assert result["demand_tier"] == "low"

    def test_ensemble_averages_scores(self) -> None:
        xgb = self._mock_model([0.6])
        lgbm = self._mock_model([0.4])
        result = predict_demand(xgb, lgbm, self._make_X())
        assert result["demand_score"] == pytest.approx(0.5, abs=0.01)

    def test_suggested_rate_positive(self) -> None:
        xgb = self._mock_model([0.7])
        result = predict_demand(xgb, None, self._make_X(), base_rate=100.0)
        assert result["suggested_rate"] > 0

    def test_per_row_scores_length(self) -> None:
        xgb = self._mock_model([0.6, 0.7, 0.8])
        result = predict_demand(xgb, None, self._make_X(3))
        assert len(result["per_row_scores"]) == 3
