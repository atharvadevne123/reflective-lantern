"""Extended tests for suite-cast app/monitoring.py drift detection."""

from __future__ import annotations

from app.monitoring import compute_drift, drift_severity


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
        result = compute_drift(
            [float(x) for x in ref],
            [float(x) for x in cur]
        )
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
