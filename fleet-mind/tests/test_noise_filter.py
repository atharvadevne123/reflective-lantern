"""Tests for fleet-mind app/noise_filter.py."""

from __future__ import annotations

import pytest

from app.noise_filter import (
    clamp_values,
    exponential_moving_average,
    median_filter,
    remove_spikes,
    simple_moving_average,
)


class TestSimpleMovingAverage:
    def test_window_one_returns_original(self) -> None:
        values = [1.0, 2.0, 3.0]
        result = simple_moving_average(values, window=1)
        assert result == pytest.approx(values)

    def test_output_length_equals_input(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert len(simple_moving_average(values, window=3)) == len(values)

    def test_converges_on_flat_signal(self) -> None:
        values = [5.0] * 10
        result = simple_moving_average(values, window=3)
        assert all(v == pytest.approx(5.0) for v in result)

    def test_empty_input(self) -> None:
        assert simple_moving_average([], window=3) == []


class TestExponentialMovingAverage:
    def test_empty_input(self) -> None:
        assert exponential_moving_average([], alpha=0.3) == []

    def test_single_value_unchanged(self) -> None:
        assert exponential_moving_average([7.0]) == pytest.approx([7.0])

    def test_output_length_equals_input(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        assert len(exponential_moving_average(values)) == 4

    def test_first_value_unchanged(self) -> None:
        values = [3.0, 10.0, 10.0]
        assert exponential_moving_average(values)[0] == pytest.approx(3.0)


class TestMedianFilter:
    def test_flat_signal_unchanged(self) -> None:
        values = [5.0] * 5
        result = median_filter(values, window=3)
        assert result == pytest.approx(values)

    def test_output_length_equals_input(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert len(median_filter(values, window=3)) == len(values)


class TestRemoveSpikes:
    def test_no_spikes_unchanged(self) -> None:
        values = [1.0, 1.1, 0.9, 1.0]
        result = remove_spikes(values, z_threshold=3.0)
        assert result == pytest.approx(values, abs=0.01)

    def test_spike_replaced_with_mean(self) -> None:
        values = [1.0] * 20 + [100.0]
        result = remove_spikes(values, z_threshold=2.0)
        assert result[-1] < 10.0


class TestClampValues:
    def test_clamps_above(self) -> None:
        assert clamp_values([5.0, 15.0], 0.0, 10.0) == pytest.approx([5.0, 10.0])

    def test_clamps_below(self) -> None:
        assert clamp_values([-1.0, 3.0], 0.0, 10.0) == pytest.approx([0.0, 3.0])

    def test_within_range_unchanged(self) -> None:
        values = [3.0, 5.0, 7.0]
        assert clamp_values(values, 0.0, 10.0) == pytest.approx(values)
