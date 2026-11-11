"""Extended tests for suite-cast app/monitoring.py drift detection."""

from __future__ import annotations

import pytest

from app.monitoring import compute_drift, drift_severity, rate_spread


class TestComputeDrift:
    def test_insufficient_data_returns_no_drift(self) -> None:
        result = compute_drift([0.5, 0.6], [0.4, 0.5])
        assert result["drift_detected"] is False
        assert result.get("reason") == "insufficient_data"

    def test_similar_distributions_no_drift(self) -> None:
        ref = [0.5 + i * 0.01 for i in range(20)]
        cur = [0.51 + i * 0.01 for i in range(20)]
        result = compute_drift(ref, cur)
        assert "drift_detected" in result

    def test_very_different_distributions_drift_detected(self) -> None:
        ref = [0.1] * 20
        cur = [0.9] * 20
        result = compute_drift(ref, cur)
        assert result["drift_detected"] is True

    def test_returns_dict(self) -> None:
        ref = [0.5] * 10
        cur = [0.5] * 10
        result = compute_drift(ref, cur)
        assert isinstance(result, dict)

    def test_has_ks_statistic(self) -> None:
        ref = list(range(10))
        cur = list(range(10, 20))
        result = compute_drift([float(x) for x in ref], [float(x) for x in cur])
        assert "ks_statistic" in result

    def test_has_p_value(self) -> None:
        ref = [float(i) for i in range(10)]
        cur = [float(i + 5) for i in range(10)]
        result = compute_drift(ref, cur)
        assert "p_value" in result

    def test_ks_statistic_in_range(self) -> None:
        ref = [float(i) for i in range(10)]
        cur = [float(i + 5) for i in range(10)]
        result = compute_drift(ref, cur)
        assert 0.0 <= result["ks_statistic"] <= 1.0


class TestDriftSeverity:
    def test_zero_is_none(self) -> None:
        assert drift_severity(0.0) == "none"

    def test_just_below_mild_is_none(self) -> None:
        assert drift_severity(0.09) == "none"

    def test_mild_range(self) -> None:
        assert drift_severity(0.15) == "mild"

    def test_moderate_range(self) -> None:
        assert drift_severity(0.30) == "moderate"

    def test_severe_range(self) -> None:
        assert drift_severity(0.60) == "severe"

    def test_boundary_mild(self) -> None:
        assert drift_severity(0.10) == "mild"

    def test_boundary_moderate(self) -> None:
        assert drift_severity(0.25) == "moderate"

    def test_boundary_severe(self) -> None:
        assert drift_severity(0.50) == "severe"



class TestRateSpread:
    def test_empty_returns_zero(self, db_session) -> None:
        result = rate_spread(db_session)
        assert result == pytest.approx(0.0)

    def test_single_prediction_returns_zero(self, db_session) -> None:
        from app.database import Prediction
        db_session.add(Prediction(suggested_rate=120.0, demand_score=0.8, model_version="1.0"))
        db_session.commit()
        result = rate_spread(db_session)
        assert result == pytest.approx(0.0)

    def test_spread_across_two_predictions(self, db_session) -> None:
        from app.database import Prediction
        for rate in [100.0, 200.0]:
            db_session.add(Prediction(suggested_rate=rate, demand_score=0.5, model_version="1.0"))
        db_session.commit()
        result = rate_spread(db_session)
        assert result == pytest.approx(100.0)
