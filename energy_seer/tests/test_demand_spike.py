"""Tests for energy_seer demand_spike module."""

from __future__ import annotations

import pytest


class TestSpikeWindows:
    def test_single_run_meets_window(self) -> None:
        from app.demand_spike import spike_windows

        values = [0.0, 5.0, 5.0, 5.0, 0.0]
        result = spike_windows(values, threshold=3.0, window=3)
        assert result == [(1, 3)]

    def test_run_too_short_excluded(self) -> None:
        from app.demand_spike import spike_windows

        values = [5.0, 5.0, 0.0, 0.0]
        result = spike_windows(values, threshold=3.0, window=3)
        assert result == []

    def test_no_spikes(self) -> None:
        from app.demand_spike import spike_windows

        result = spike_windows([1.0, 2.0, 1.0], threshold=5.0, window=1)
        assert result == []

    def test_entire_series_is_spike(self) -> None:
        from app.demand_spike import spike_windows

        result = spike_windows([10.0, 10.0, 10.0], threshold=5.0, window=2)
        assert result == [(0, 2)]


class TestDemandPercentile:
    def test_median(self) -> None:
        from app.demand_spike import demand_percentile

        result = demand_percentile([1.0, 2.0, 3.0, 4.0, 5.0], percentile=50.0)
        assert result == pytest.approx(3.0, abs=0.01)

    def test_100th_percentile(self) -> None:
        from app.demand_spike import demand_percentile

        assert demand_percentile([1.0, 5.0, 3.0], percentile=100.0) == pytest.approx(5.0)

    def test_empty_raises(self) -> None:
        from app.demand_spike import demand_percentile

        with pytest.raises(ValueError):
            demand_percentile([])

    def test_out_of_range_raises(self) -> None:
        from app.demand_spike import demand_percentile

        with pytest.raises(ValueError):
            demand_percentile([1.0, 2.0], percentile=0.0)


class TestSmoothDemand:
    def test_same_length(self) -> None:
        from app.demand_spike import smooth_demand

        result = smooth_demand([1.0, 2.0, 3.0])
        assert len(result) == 3

    def test_first_element_unchanged(self) -> None:
        from app.demand_spike import smooth_demand

        result = smooth_demand([5.0, 1.0, 1.0], alpha=0.5)
        assert result[0] == pytest.approx(5.0)

    def test_empty_raises(self) -> None:
        from app.demand_spike import smooth_demand

        with pytest.raises(ValueError):
            smooth_demand([])

    def test_invalid_alpha_raises(self) -> None:
        from app.demand_spike import smooth_demand

        with pytest.raises(ValueError):
            smooth_demand([1.0, 2.0], alpha=0.0)


class TestSustainedDemandViolation:
    def test_single_violation_detected(self) -> None:
        from app.demand_spike import sustained_demand_violation

        values = [1.0, 5.0, 5.0, 5.0, 1.0]
        result = sustained_demand_violation(values, threshold=3.0, min_duration=3)
        assert len(result) == 1
        assert result[0]["start"] == 1
        assert result[0]["duration"] == 3

    def test_too_short_not_reported(self) -> None:
        from app.demand_spike import sustained_demand_violation

        values = [5.0, 5.0, 1.0, 1.0]
        result = sustained_demand_violation(values, threshold=3.0, min_duration=3)
        assert result == []

    def test_empty_input(self) -> None:
        from app.demand_spike import sustained_demand_violation

        assert sustained_demand_violation([], threshold=2.0) == []

    @pytest.mark.parametrize("min_dur,expected", [(1, 1), (2, 1), (4, 0)])
    def test_min_duration_boundary(self, min_dur: int, expected: int) -> None:
        from app.demand_spike import sustained_demand_violation

        values = [0.0, 6.0, 6.0, 6.0, 0.0]
        result = sustained_demand_violation(values, threshold=5.0, min_duration=min_dur)
        assert len(result) == expected


class TestDemandVariability:
    def test_uniform_series_zero_std(self) -> None:
        from app.demand_spike import demand_variability

        result = demand_variability([5.0, 5.0, 5.0])
        assert result["std"] == pytest.approx(0.0, abs=1e-9)
        assert result["range"] == pytest.approx(0.0)

    def test_returns_all_keys(self) -> None:
        from app.demand_spike import demand_variability

        result = demand_variability([1.0, 2.0, 3.0, 4.0])
        assert {"cv", "range", "iqr", "std"} <= result.keys()

    def test_insufficient_data(self) -> None:
        from app.demand_spike import demand_variability

        result = demand_variability([1.0])
        assert result == {"cv": 0.0, "range": 0.0, "iqr": 0.0, "std": 0.0}
