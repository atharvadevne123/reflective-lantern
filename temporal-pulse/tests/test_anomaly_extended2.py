"""Extended anomaly utility tests for temporal-pulse with parametrize and boundary coverage."""

from __future__ import annotations

import pytest

from app.anomaly import (
    anomaly_burst_count,
    anomaly_density,
    consecutive_anomaly_lengths,
    score_percentile_threshold,
)


@pytest.mark.parametrize(
    "percentile,expected",
    [
        (100.0, 10.0),
        (0.0, 1.0),
        (50.0, 5.5),
    ],
)
def test_score_percentile_threshold_values(percentile: float, expected: float) -> None:
    scores = list(range(1, 11))
    result = score_percentile_threshold(scores, percentile=percentile)
    assert result == pytest.approx(expected, abs=0.5)


def test_score_percentile_threshold_empty_returns_zero() -> None:
    assert score_percentile_threshold([]) == 0.0


@pytest.mark.parametrize("percentile", [25.0, 50.0, 75.0, 90.0, 95.0, 99.0])
def test_score_percentile_threshold_in_range(percentile: float) -> None:
    scores = list(range(1, 101))
    result = score_percentile_threshold(scores, percentile=percentile)
    assert min(scores) <= result <= max(scores)


def test_consecutive_anomaly_lengths_empty_flags() -> None:
    assert consecutive_anomaly_lengths([]) == []


def test_consecutive_anomaly_lengths_no_anomalies() -> None:
    flags = [False, False, False, False]
    assert consecutive_anomaly_lengths(flags) == []


def test_consecutive_anomaly_lengths_all_anomalies() -> None:
    flags = [True, True, True, True]
    assert consecutive_anomaly_lengths(flags) == [4]


@pytest.mark.parametrize(
    "flags,expected",
    [
        ([True, False, True], [1, 1]),
        ([True, True, False, True, True, True], [2, 3]),
        ([False, True, True, False, True], [2, 1]),
        ([True], [1]),
        ([False], []),
    ],
)
def test_consecutive_anomaly_lengths_patterns(flags: list[bool], expected: list[int]) -> None:
    assert consecutive_anomaly_lengths(flags) == expected


def test_consecutive_anomaly_lengths_trailing_true() -> None:
    flags = [False, True, True]
    result = consecutive_anomaly_lengths(flags)
    assert result == [2]


def test_anomaly_density_empty_returns_empty() -> None:
    assert anomaly_density([], threshold=0.5) == []


def test_anomaly_density_all_above_threshold() -> None:
    scores = [1.0, 1.0, 1.0, 1.0, 1.0]
    result = anomaly_density(scores, threshold=0.5, window=3)
    assert all(d == pytest.approx(1.0, abs=1e-4) for d in result)


def test_anomaly_density_all_below_threshold() -> None:
    scores = [0.1, 0.1, 0.1, 0.1]
    result = anomaly_density(scores, threshold=0.5, window=3)
    assert all(d == pytest.approx(0.0, abs=1e-4) for d in result)


def test_anomaly_density_length_matches_input() -> None:
    scores = [float(i) for i in range(10)]
    result = anomaly_density(scores, threshold=5.0, window=3)
    assert len(result) == 10


@pytest.mark.parametrize("window", [1, 3, 5, 10])
def test_anomaly_density_density_in_zero_one(window: int) -> None:
    import random

    random.seed(42)
    scores = [random.random() for _ in range(20)]
    result = anomaly_density(scores, threshold=0.5, window=window)
    assert all(0.0 <= d <= 1.0 for d in result)


def test_anomaly_burst_count_no_bursts() -> None:
    scores = [0.1, 0.2, 0.1, 0.0]
    assert anomaly_burst_count(scores, threshold=0.5) == 0


def test_anomaly_burst_count_single_long_burst() -> None:
    scores = [0.1, 0.9, 0.9, 0.9, 0.1]
    assert anomaly_burst_count(scores, threshold=0.5, min_burst=3) == 1


def test_anomaly_burst_count_min_burst_not_reached() -> None:
    scores = [0.1, 0.9, 0.9, 0.1]
    assert anomaly_burst_count(scores, threshold=0.5, min_burst=3) == 0


def test_anomaly_burst_count_multiple_bursts() -> None:
    scores = [0.9, 0.9, 0.9, 0.0, 0.9, 0.9, 0.9]
    assert anomaly_burst_count(scores, threshold=0.5, min_burst=2) == 2


def test_anomaly_burst_count_empty_input() -> None:
    assert anomaly_burst_count([], threshold=0.5) == 0
