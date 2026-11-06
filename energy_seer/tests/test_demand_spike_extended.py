"""Extended tests for app/demand_spike.py."""

from __future__ import annotations

import pytest

from app.demand_spike import (
    consecutive_spike_run,
    demand_percentile,
    demand_trend,
    demand_variability,
    detect_spike,
    normalise_demand,
    rolling_spike_count,
    smooth_demand,
    spike_ratio,
    spike_severity,
    spike_windows,
    sustained_demand_violation,
)


class TestDetectSpikeEdgeCases:
    def test_empty_list_returns_minimal_dict(self) -> None:
        result = detect_spike([])
        assert result["spike_count"] == 0

    def test_single_value_insufficient(self) -> None:
        result = detect_spike([5.0])
        assert result.get("method") == "insufficient_data"

    def test_all_identical_values(self) -> None:
        result = detect_spike([3.0] * 10)
        assert result["spike_count"] == 0

    def test_clear_spike_detected(self) -> None:
        values = [1.0] * 20 + [100.0]
        result = detect_spike(values, z_threshold=2.5)
        assert result["spike_count"] >= 1
        assert 20 in result["spike_indices"]

    @pytest.mark.parametrize("threshold", [1.5, 2.0, 3.0])
    def test_higher_threshold_fewer_spikes(self, threshold: float) -> None:
        values = [1.0] * 10 + [5.0, 10.0, 15.0]
        low = detect_spike(values, z_threshold=1.5)
        high = detect_spike(values, z_threshold=3.0)
        assert low["spike_count"] >= high["spike_count"]


class TestSpikeSeverity:
    @pytest.mark.parametrize(
        "value,mean,std,expected",
        [
            (1.0, 1.0, 0.1, "normal"),
            (5.0, 1.0, 1.0, "moderate"),
            (7.0, 1.0, 1.0, "high"),
            (12.0, 1.0, 1.0, "critical"),
        ],
    )
    def test_severity_levels(
        self, value: float, mean: float, std: float, expected: str
    ) -> None:
        assert spike_severity(value, mean, std, z_threshold=2.5) == expected

    def test_zero_std_returns_normal(self) -> None:
        assert spike_severity(100.0, 50.0, 0.0) == "normal"


class TestRollingSpike:
    def test_returns_same_length(self) -> None:
        values = [1.0] * 20
        result = rolling_spike_count(values, window=5)
        assert len(result) == 20

    def test_all_zero_for_flat_series(self) -> None:
        values = [3.0] * 30
        result = rolling_spike_count(values, window=10)
        assert all(c == 0 for c in result)


class TestNormaliseDemand:
    def test_all_zeros_for_constant(self) -> None:
        assert normalise_demand([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]

    def test_empty_list_returns_empty(self) -> None:
        assert normalise_demand([]) == []

    def test_range_is_zero_to_one(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        norm = normalise_demand(values)
        assert min(norm) == pytest.approx(0.0)
        assert max(norm) == pytest.approx(1.0)


class TestSustainedViolation:
    def test_no_violation_below_threshold(self) -> None:
        assert sustained_demand_violation([1.0, 2.0, 3.0], threshold=10.0) == []

    def test_detects_long_violation(self) -> None:
        values = [10.0] * 5 + [1.0] * 5
        result = sustained_demand_violation(values, threshold=5.0, min_duration=3)
        assert len(result) == 1
        assert result[0]["duration"] == 5

    def test_short_violation_not_reported(self) -> None:
        values = [10.0, 10.0, 1.0]
        result = sustained_demand_violation(values, threshold=5.0, min_duration=3)
        assert result == []


class TestDemandVariability:
    def test_constant_series_zero_cv(self) -> None:
        result = demand_variability([5.0] * 10)
        assert result["cv"] == pytest.approx(0.0)
        assert result["range"] == pytest.approx(0.0)

    def test_insufficient_data(self) -> None:
        result = demand_variability([1.0])
        assert result["cv"] == 0.0

    def test_varied_series_has_positive_cv(self) -> None:
        result = demand_variability([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["cv"] > 0.0


class TestDemandTrend:
    def test_rising_trend(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        result = demand_trend(values)
        assert result["trend"] == "rising"

    def test_falling_trend(self) -> None:
        values = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        result = demand_trend(values)
        assert result["trend"] == "falling"

    def test_flat_trend(self) -> None:
        values = [5.0] * 8
        result = demand_trend(values)
        assert result["trend"] == "flat"

    def test_insufficient_data_returns_flat(self) -> None:
        result = demand_trend([1.0, 2.0])
        assert result["trend"] == "flat"


class TestSmoothDemand:
    def test_constant_series_unchanged(self) -> None:
        values = [5.0] * 5
        result = smooth_demand(values, alpha=0.3)
        assert all(r == pytest.approx(5.0) for r in result)

    def test_same_length_as_input(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert len(smooth_demand(values)) == len(values)

    def test_first_value_preserved(self) -> None:
        values = [10.0, 1.0, 1.0, 1.0]
        assert smooth_demand(values, alpha=0.5)[0] == pytest.approx(10.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            smooth_demand([])

    @pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
    def test_valid_alpha_accepted(self, alpha: float) -> None:
        result = smooth_demand([1.0, 2.0, 3.0], alpha=alpha)
        assert len(result) == 3


class TestDemandPercentile:
    def test_p50_is_median(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert demand_percentile(values, 50.0) == pytest.approx(3.0)

    def test_p100_is_max(self) -> None:
        values = [1.0, 2.0, 3.0]
        assert demand_percentile(values, 100.0) == pytest.approx(3.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            demand_percentile([])

    def test_invalid_percentile_raises(self) -> None:
        with pytest.raises(ValueError):
            demand_percentile([1.0, 2.0], 0.0)


class TestSpikeWindows:
    def test_no_spikes_returns_empty(self) -> None:
        assert spike_windows([1.0, 2.0, 3.0], threshold=10.0) == []

    def test_long_spike_window_returned(self) -> None:
        values = [10.0] * 5
        result = spike_windows(values, threshold=5.0, window=3)
        assert len(result) == 1
        assert result[0] == (0, 4)


class TestConsecutiveSpikeRun:
    def test_no_spikes_returns_zero(self) -> None:
        assert consecutive_spike_run([1.0] * 10) == 0

    def test_all_spikes_returns_full_length(self) -> None:
        values = [1.0] * 5 + [100.0] * 10 + [1.0] * 5
        run = consecutive_spike_run(values, z_threshold=2.0)
        assert run >= 5


class TestSpikeRatio:
    def test_no_spikes_returns_zero(self) -> None:
        assert spike_ratio([1.0] * 20) == pytest.approx(0.0)

    def test_all_spikes_returns_nonzero(self) -> None:
        values = [1.0] * 15 + [100.0, 200.0, 300.0]
        ratio = spike_ratio(values, z_threshold=1.5)
        assert ratio > 0.0

    def test_single_value_returns_zero(self) -> None:
        assert spike_ratio([5.0]) == pytest.approx(0.0)
