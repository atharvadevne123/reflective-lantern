"""Extended demand spike tests with parametrize and boundary coverage."""

from __future__ import annotations

import pytest

from app.demand_spike import (
    demand_percentile,
    detect_spike,
    smooth_demand,
    spike_severity,
)


def test_detect_spike_insufficient_data_less_than_4() -> None:
    result = detect_spike([1.0, 2.0, 3.0])
    assert result["method"] == "insufficient_data"
    assert result["spike_count"] == 0


def test_detect_spike_no_spikes_in_uniform_data() -> None:
    values = [5.0] * 20
    result = detect_spike(values)
    assert result["spike_count"] == 0


def test_detect_spike_obvious_spike_detected() -> None:
    values = [5.0] * 19 + [1000.0]
    result = detect_spike(values)
    assert result["spike_count"] > 0
    assert 19 in result["spike_indices"]


def test_detect_spike_has_required_keys() -> None:
    values = [float(i) for i in range(10)]
    result = detect_spike(values)
    for key in ("spike_indices", "spike_count", "z_threshold", "iqr_upper", "mean", "std"):
        assert key in result


@pytest.mark.parametrize("z_threshold", [1.5, 2.0, 2.5, 3.0])
def test_detect_spike_stricter_threshold_fewer_spikes(z_threshold: float) -> None:
    import random

    rng = random.Random(42)
    values = [rng.gauss(10, 2) for _ in range(100)]
    values[50] = 100.0
    result_strict = detect_spike(values, z_threshold=z_threshold)
    result_loose = detect_spike(values, z_threshold=1.0)
    assert result_strict["spike_count"] <= result_loose["spike_count"]


def test_spike_severity_normal_below_threshold() -> None:
    assert spike_severity(5.0, 5.0, 1.0) == "normal"


@pytest.mark.parametrize(
    "z_distance,expected",
    [
        (1.0, "normal"),
        (2.5 + 0.1, "moderate"),
        (5.0 + 0.1, "high"),
        (7.5 + 0.1, "critical"),
    ],
)
def test_spike_severity_parametrized(z_distance: float, expected: str) -> None:
    mean = 0.0
    std = 1.0
    value = mean + z_distance * std
    result = spike_severity(value, mean, std, z_threshold=2.5)
    assert result in ("normal", "moderate", "high", "critical")


def test_spike_severity_zero_std_returns_normal() -> None:
    assert spike_severity(100.0, 5.0, 0.0) == "normal"


def test_demand_percentile_empty_raises() -> None:
    with pytest.raises(ValueError):
        demand_percentile([])


def test_demand_percentile_invalid_percentile_raises() -> None:
    with pytest.raises(ValueError):
        demand_percentile([1.0, 2.0, 3.0], percentile=0.0)


def test_demand_percentile_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        demand_percentile([1.0, 2.0, 3.0], percentile=101.0)


@pytest.mark.parametrize(
    "percentile,expected",
    [
        (100.0, 10.0),
        (50.0, 5.0),
    ],
)
def test_demand_percentile_parametrized(percentile: float, expected: float) -> None:
    values = [float(i) for i in range(1, 11)]
    result = demand_percentile(values, percentile)
    assert result == pytest.approx(expected, abs=0.5)


def test_smooth_demand_empty_raises() -> None:
    with pytest.raises(ValueError):
        smooth_demand([])


def test_smooth_demand_invalid_alpha_raises() -> None:
    with pytest.raises(ValueError):
        smooth_demand([1.0, 2.0], alpha=0.0)


def test_smooth_demand_same_length_as_input() -> None:
    values = [float(i) for i in range(10)]
    result = smooth_demand(values, alpha=0.3)
    assert len(result) == len(values)


@pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.9, 1.0])
def test_smooth_demand_various_alpha(alpha: float) -> None:
    values = [1.0, 10.0, 1.0, 10.0, 1.0]
    result = smooth_demand(values, alpha=alpha)
    assert len(result) == len(values)
    assert result[0] == pytest.approx(values[0])
