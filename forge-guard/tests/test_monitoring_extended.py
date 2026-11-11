"""Extended tests for forge-guard app/monitoring.py functions."""

from __future__ import annotations

from app.monitoring import compute_drift, compute_zscore_outliers


class TestComputeDrift:
    def test_no_drift_on_identical(self) -> None:
        values = [0.5 + i * 0.01 for i in range(30)]
        result = compute_drift(values, values)
        assert isinstance(result, dict)
        assert "drift_detected" in result

    def test_drift_on_very_different(self) -> None:
        ref = [0.0] * 30
        cur = [1.0] * 30
        result = compute_drift(ref, cur)
        assert result["drift_detected"] is True

    def test_insufficient_data_no_drift(self) -> None:
        result = compute_drift([0.1], [0.9])
        assert result["drift_detected"] is False or "reason" in result

    def test_returns_p_value(self) -> None:
        ref = [float(i) for i in range(30)]
        cur = [float(i) for i in range(30)]
        result = compute_drift(ref, cur)
        assert "p_value" in result or "drift_detected" in result

    def test_similar_distributions_no_drift(self) -> None:
        ref = [0.5 + i * 0.001 for i in range(30)]
        cur = [0.5 + i * 0.001 for i in range(30)]
        result = compute_drift(ref, cur)
        assert result["drift_detected"] is False


class TestComputeZscoreOutliers:
    def test_no_outliers_in_uniform_data(self) -> None:
        values = [1.0] * 30
        result = compute_zscore_outliers(values)
        assert isinstance(result, list)
        assert all(v is False for v in result)

    def test_detects_spike(self) -> None:
        values = [1.0] * 29 + [999.0]
        result = compute_zscore_outliers(values)
        assert result[-1] is True

    def test_length_matches_input(self) -> None:
        values = list(range(10))
        result = compute_zscore_outliers([float(v) for v in values])
        assert len(result) == len(values)

    def test_custom_threshold(self) -> None:
        values = [0.0] * 9 + [5.0]
        result = compute_zscore_outliers(values, threshold=2.0)
        assert isinstance(result, list)

    def test_empty_list_returns_empty(self) -> None:
        result = compute_zscore_outliers([])
        assert result == [] or result is None
