"""Extended anomaly analysis tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.anomaly import (
    anomaly_rate,
    anomaly_summary,
    batch_compute_severity,
    compute_severity,
    consecutive_anomaly_runs,
    ewma_smooth,
    flag_anomaly_rate,
    iqr_flag,
    rolling_anomaly_flag,
    zscore_flag,
)


def test_zscore_normal() -> None:
    assert not zscore_flag(10.0, mean=10.0, std=2.0, threshold=3.0)


def test_zscore_outlier() -> None:
    assert zscore_flag(20.0, mean=10.0, std=1.0, threshold=3.0)


def test_zscore_zero_std() -> None:
    assert not zscore_flag(5.0, mean=5.0, std=0.0)


def test_iqr_within_fence() -> None:
    assert not iqr_flag(10.0, q1=8.0, q3=12.0)


def test_iqr_outside_fence() -> None:
    assert iqr_flag(30.0, q1=8.0, q3=12.0)


def test_iqr_below_fence() -> None:
    assert iqr_flag(-5.0, q1=8.0, q3=12.0)


def test_compute_severity_none() -> None:
    ref = list(np.random.default_rng(1).normal(10, 1, 200))
    result = compute_severity(10.0, ref)
    assert result["severity"] == "none"


def test_compute_severity_critical() -> None:
    ref = list(np.random.default_rng(1).normal(10, 1, 200))
    result = compute_severity(100.0, ref)
    assert result["severity"] == "critical"
    assert result["z_flag"] is True
    assert result["iqr_flag"] is True


@pytest.mark.parametrize("value,expected", [(10.0, "none"), (100.0, "critical")])
def test_compute_severity_parametrized(value, expected) -> None:
    ref = list(np.random.default_rng(42).normal(10, 1, 100))
    result = compute_severity(value, ref)
    assert result["severity"] == expected


def test_compute_severity_returns_required_keys() -> None:
    ref = list(np.random.default_rng(0).normal(5, 1, 50))
    result = compute_severity(5.0, ref)
    assert "z_flag" in result
    assert "iqr_flag" in result
    assert "severity" in result


def test_zscore_exactly_at_threshold() -> None:
    assert not zscore_flag(13.0, mean=10.0, std=1.0, threshold=3.0)


@pytest.mark.parametrize("threshold", [1.0, 2.0, 3.0, 4.0])
def test_zscore_various_thresholds(threshold) -> None:
    assert zscore_flag(10.0 + threshold * 1.5, mean=10.0, std=1.0, threshold=threshold)


def test_iqr_exactly_at_fence() -> None:
    assert not iqr_flag(20.0, q1=8.0, q3=12.0, k=2.0)


def test_compute_severity_warning_only_one_flag() -> None:
    ref = list(np.random.default_rng(7).normal(10, 1, 200))
    # Use a modest outlier that might trigger only one of the two tests
    result = compute_severity(13.5, ref)
    assert result["severity"] in ("none", "warning", "critical")


def test_batch_compute_severity_basic() -> None:
    ref = list(np.random.default_rng(1).normal(10, 1, 200))
    results = batch_compute_severity([10.0, 100.0], ref)
    assert len(results) == 2
    assert results[0]["severity"] == "none"
    assert results[1]["severity"] == "critical"


def test_batch_compute_severity_includes_value_key() -> None:
    ref = list(np.random.default_rng(2).normal(10, 1, 100))
    results = batch_compute_severity([10.0, 11.0], ref)
    assert all("value" in r for r in results)
    assert results[0]["value"] == 10.0


def test_batch_compute_severity_small_reference() -> None:
    results = batch_compute_severity([5.0, 6.0], [1.0, 2.0, 3.0])
    assert all(r["severity"] == "none" for r in results)


def test_batch_compute_severity_empty_input() -> None:
    ref = list(np.random.default_rng(3).normal(10, 1, 100))
    results = batch_compute_severity([], ref)
    assert results == []


@pytest.mark.parametrize(
    "flagged,total,expected",
    [
        ([], 0, 0.0),
        ([{"severity": "none"}], 1, 0.0),
        ([{"severity": "warning"}], 1, 1.0),
        ([{"severity": "critical"}, {"severity": "none"}], 2, 0.5),
    ],
)
def test_anomaly_rate_parametrized(flagged, total, expected) -> None:
    assert anomaly_rate(flagged) == pytest.approx(expected)


def test_top_anomalies_returns_critical_first() -> None:
    from app.anomaly import top_anomalies

    data = [
        {"severity": "none", "value": 10.0},
        {"severity": "critical", "value": 100.0},
        {"severity": "warning", "value": 15.0},
    ]
    result = top_anomalies(data, n=3)
    assert result[0]["severity"] == "critical"
    assert result[1]["severity"] == "warning"


def test_top_anomalies_respects_n() -> None:
    from app.anomaly import top_anomalies

    data = [{"severity": "critical", "value": float(i)} for i in range(20)]
    result = top_anomalies(data, n=5)
    assert len(result) == 5


def test_top_anomalies_empty() -> None:
    from app.anomaly import top_anomalies

    assert top_anomalies([], n=5) == []


def test_top_anomalies_custom_order() -> None:
    from app.anomaly import top_anomalies

    data = [
        {"severity": "none", "value": 1.0},
        {"severity": "warning", "value": 2.0},
    ]
    result = top_anomalies(data, n=2, severity_order=["warning", "critical", "none"])
    assert result[0]["severity"] == "warning"


def test_compute_percentile_bounds_basic() -> None:
    from app.anomaly import compute_percentile_bounds

    ref = list(range(1, 101))  # 1..100
    bounds = compute_percentile_bounds(ref, lower_pct=1.0, upper_pct=99.0)
    assert bounds["lower"] < bounds["upper"]
    assert bounds["median"] == pytest.approx(50.5, rel=1e-2)


def test_compute_percentile_bounds_keys() -> None:
    from app.anomaly import compute_percentile_bounds

    bounds = compute_percentile_bounds([5.0] * 20)
    assert set(bounds.keys()) == {"lower", "upper", "median", "mean"}


def test_compute_percentile_bounds_empty_raises() -> None:
    from app.anomaly import compute_percentile_bounds

    with pytest.raises(ValueError, match="non-empty"):
        compute_percentile_bounds([])


def test_compute_percentile_bounds_invalid_pct_raises() -> None:
    from app.anomaly import compute_percentile_bounds

    with pytest.raises(ValueError):
        compute_percentile_bounds([1.0, 2.0], lower_pct=80.0, upper_pct=20.0)


@pytest.mark.parametrize("lower,upper", [(0, 100), (5, 95), (10, 90), (25, 75)])
def test_compute_percentile_bounds_parametrized(lower, upper) -> None:
    from app.anomaly import compute_percentile_bounds

    ref = list(np.random.default_rng(1).normal(10, 2, 200))
    bounds = compute_percentile_bounds(ref, lower_pct=lower, upper_pct=upper)
    assert bounds["lower"] <= bounds["upper"]


def test_classify_consumption_low() -> None:
    from app.anomaly import classify_consumption

    assert classify_consumption(3.0, low_threshold=5.0, high_threshold=15.0) == "low"


def test_classify_consumption_normal() -> None:
    from app.anomaly import classify_consumption

    assert classify_consumption(10.0, low_threshold=5.0, high_threshold=15.0) == "normal"


def test_classify_consumption_high() -> None:
    from app.anomaly import classify_consumption

    assert classify_consumption(20.0, low_threshold=5.0, high_threshold=15.0) == "high"


def test_classify_consumption_boundary_low() -> None:
    from app.anomaly import classify_consumption

    assert classify_consumption(5.0, low_threshold=5.0, high_threshold=15.0) == "normal"


def test_classify_consumption_boundary_high() -> None:
    from app.anomaly import classify_consumption

    assert classify_consumption(15.0, low_threshold=5.0, high_threshold=15.0) == "high"


def test_classify_consumption_invalid_thresholds() -> None:
    from app.anomaly import classify_consumption

    with pytest.raises(ValueError):
        classify_consumption(10.0, low_threshold=20.0, high_threshold=10.0)


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, "low"),
        (4.9, "low"),
        (5.0, "normal"),
        (14.9, "normal"),
        (15.0, "high"),
        (100.0, "high"),
    ],
)
def test_classify_consumption_parametrized(value, expected) -> None:
    from app.anomaly import classify_consumption

    assert classify_consumption(value, low_threshold=5.0, high_threshold=15.0) == expected


def test_compute_z_score_basic() -> None:
    from app.anomaly import compute_z_score

    assert compute_z_score(10.0, mean=10.0, std=1.0) == pytest.approx(0.0)


def test_compute_z_score_positive() -> None:
    from app.anomaly import compute_z_score

    assert compute_z_score(13.0, mean=10.0, std=1.0) == pytest.approx(3.0)


def test_compute_z_score_negative() -> None:
    from app.anomaly import compute_z_score

    assert compute_z_score(7.0, mean=10.0, std=1.0) == pytest.approx(-3.0)


def test_compute_z_score_zero_std() -> None:
    from app.anomaly import compute_z_score

    assert compute_z_score(5.0, mean=5.0, std=0.0) == pytest.approx(0.0)


def test_flag_z_score_outliers_finds_outlier() -> None:
    from app.anomaly import flag_z_score_outliers

    data = [10.0] * 50 + [1000.0]
    outliers = flag_z_score_outliers(data)
    assert 50 in outliers


def test_flag_z_score_outliers_none_on_flat() -> None:
    from app.anomaly import flag_z_score_outliers

    data = [5.0] * 20
    assert flag_z_score_outliers(data) == []


def test_flag_z_score_outliers_empty_raises() -> None:
    from app.anomaly import flag_z_score_outliers

    with pytest.raises(ValueError):
        flag_z_score_outliers([])


def test_flag_z_score_outliers_bad_threshold_raises() -> None:
    from app.anomaly import flag_z_score_outliers

    with pytest.raises(ValueError):
        flag_z_score_outliers([1.0, 2.0], threshold=0.0)


@pytest.mark.parametrize("threshold", [1.0, 2.0, 3.0])
def test_flag_z_score_outliers_custom_threshold(threshold) -> None:
    from app.anomaly import flag_z_score_outliers

    data = [10.0] * 40 + [10000.0]
    result = flag_z_score_outliers(data, threshold=threshold)
    assert len(result) >= 1


def test_top_anomalies_returns_most_severe_first() -> None:
    from app.anomaly import top_anomalies

    data = [
        {"severity": "none", "value": 5.0},
        {"severity": "critical", "value": 100.0},
        {"severity": "warning", "value": 50.0},
    ]
    result = top_anomalies(data, n=3)
    assert result[0]["severity"] == "critical"
    assert result[1]["severity"] == "warning"


def test_top_anomalies_limits_to_n() -> None:
    from app.anomaly import top_anomalies

    data = [{"severity": "critical", "value": float(i)} for i in range(20)]
    result = top_anomalies(data, n=5)
    assert len(result) == 5


def test_top_anomalies_empty_input() -> None:
    from app.anomaly import top_anomalies

    assert top_anomalies([]) == []


def test_top_anomalies_custom_order_warning_first() -> None:
    from app.anomaly import top_anomalies

    data = [
        {"severity": "warning", "value": 5.0},
        {"severity": "critical", "value": 10.0},
    ]
    result = top_anomalies(data, n=2, severity_order=["warning", "critical", "none"])
    assert result[0]["severity"] == "warning"


@pytest.mark.parametrize("n", [1, 5, 10])
def test_top_anomalies_n_parametrized(n) -> None:
    from app.anomaly import top_anomalies

    data = [{"severity": "warning", "value": float(i)} for i in range(15)]
    result = top_anomalies(data, n=n)
    assert len(result) <= n


def test_flag_anomaly_rate_all_anomalies() -> None:
    assert flag_anomaly_rate([True, True, True]) == pytest.approx(1.0)


def test_flag_anomaly_rate_no_anomalies() -> None:
    assert flag_anomaly_rate([False, False, False]) == pytest.approx(0.0)


def test_flag_anomaly_rate_mixed() -> None:
    assert flag_anomaly_rate([True, False, True, False]) == pytest.approx(0.5)


def test_flag_anomaly_rate_empty() -> None:
    assert flag_anomaly_rate([]) == 0.0


def test_consecutive_anomaly_runs_single_run() -> None:
    flags = [False, True, True, True, False]
    runs = consecutive_anomaly_runs(flags)
    assert runs == [(1, 3)]


def test_consecutive_anomaly_runs_multiple() -> None:
    flags = [True, False, True, True]
    runs = consecutive_anomaly_runs(flags)
    assert (0, 0) in runs
    assert (2, 3) in runs


def test_consecutive_anomaly_runs_none() -> None:
    flags = [False, False, False]
    assert consecutive_anomaly_runs(flags) == []


def test_consecutive_anomaly_runs_all_true() -> None:
    flags = [True, True, True]
    runs = consecutive_anomaly_runs(flags)
    assert runs == [(0, 2)]


@pytest.mark.parametrize("rate,expected", [(1.0, True), (0.0, False)])
def test_flag_anomaly_rate_boundary(rate, expected) -> None:
    n = 10
    flags = [True] * int(n * rate) + [False] * (n - int(n * rate))
    result = flag_anomaly_rate(flags)
    assert (result == pytest.approx(1.0)) == expected


@pytest.mark.parametrize(
    "value,mean,std,threshold,expected",
    [
        (10.0, 10.0, 2.0, 3.0, False),  # within threshold
        (19.0, 10.0, 2.0, 3.0, True),  # 4.5 stdev away
        (10.0, 10.0, 0.0, 3.0, False),  # zero std -> not anomaly
        (100.0, 10.0, 1.0, 3.0, True),  # far outlier
    ],
)
def test_zscore_flag_parametrized(value: float, mean: float, std: float, threshold: float, expected: bool) -> None:
    assert zscore_flag(value, mean=mean, std=std, threshold=threshold) == expected


@pytest.mark.parametrize(
    "value,q1,q3,expected",
    [
        (10.0, 8.0, 12.0, False),  # within fence
        (30.0, 8.0, 12.0, True),  # above upper fence
        (-5.0, 8.0, 12.0, True),  # below lower fence
        (12.0, 8.0, 12.0, False),  # exactly at q3
    ],
)
def test_iqr_flag_parametrized(value: float, q1: float, q3: float, expected: bool) -> None:
    assert iqr_flag(value, q1=q1, q3=q3) == expected


def test_compute_severity_small_reference() -> None:
    result = compute_severity(10.0, reference=[5.0, 6.0])
    assert "severity" in result


def test_batch_compute_severity_empty() -> None:
    result = batch_compute_severity([], reference=[10.0, 11.0, 12.0, 9.0, 10.5])
    assert result == []


def test_anomaly_rate_all_critical() -> None:
    severities = [{"severity": "critical", "z_flagged": True, "iqr_flagged": True} for _ in range(10)]
    assert anomaly_rate(severities) == pytest.approx(1.0)


def test_anomaly_rate_all_none() -> None:
    severities = [{"severity": "none", "z_flagged": False, "iqr_flagged": False} for _ in range(10)]
    assert anomaly_rate(severities) == pytest.approx(0.0)


@pytest.mark.parametrize("n_values,n_ref", [(1, 10), (5, 20), (100, 50)])
def test_batch_compute_severity_sizes(n_values: int, n_ref: int) -> None:
    rng = np.random.default_rng(99)
    ref = list(rng.normal(10, 1, n_ref).tolist())
    vals = list(rng.normal(10, 1, n_values).tolist())
    results = batch_compute_severity(vals, reference=ref)
    assert len(results) == n_values
    for r in results:
        assert r["severity"] in ("none", "warning", "critical")


def test_anomaly_rate_mixed_severities() -> None:
    severities = [
        {"severity": "critical"},
        {"severity": "warning"},
        {"severity": "none"},
        {"severity": "critical"},
    ]
    rate = anomaly_rate(severities)
    assert rate == pytest.approx(0.75, rel=1e-4)


def test_compute_severity_warning_zone() -> None:
    ref = [10.0] * 200
    # value exactly at 4 std from mean (std ~0 so any deviation is flagged)
    result = compute_severity(10.5, ref, z_threshold=3.0)
    assert "severity" in result


def test_ewma_smooth_length() -> None:
    result = ewma_smooth([1.0, 2.0, 3.0, 4.0], alpha=0.3)
    assert len(result) == 4


def test_ewma_smooth_first_value_unchanged() -> None:
    values = [10.0, 20.0, 30.0]
    result = ewma_smooth(values, alpha=0.5)
    assert result[0] == pytest.approx(10.0)


def test_ewma_smooth_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        ewma_smooth([])


def test_ewma_smooth_invalid_alpha_raises() -> None:
    with pytest.raises(ValueError, match="alpha"):
        ewma_smooth([1.0, 2.0], alpha=0.0)


@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
def test_ewma_smooth_monotone_input(alpha: float) -> None:
    values = [float(i) for i in range(1, 11)]
    result = ewma_smooth(values, alpha=alpha)
    assert len(result) == len(values)
    assert all(isinstance(v, float) for v in result)


def test_compute_severity_no_anomaly_returns_none() -> None:
    ref = [10.0] * 20
    result = compute_severity(10.0, ref)
    assert result["severity"] == "none"


def test_compute_severity_result_has_required_keys() -> None:
    ref = [float(i) for i in range(1, 21)]
    result = compute_severity(50.0, ref)
    assert "severity" in result
    assert "z_flag" in result


def test_classify_consumption_low_inline() -> None:
    from app.anomaly import classify_consumption

    assert classify_consumption(5.0, 10.0, 20.0) == "low"


def test_classify_consumption_normal_inline() -> None:
    from app.anomaly import classify_consumption

    assert classify_consumption(15.0, 10.0, 20.0) == "normal"


def test_classify_consumption_high_inline() -> None:
    from app.anomaly import classify_consumption

    assert classify_consumption(25.0, 10.0, 20.0) == "high"


def test_anomaly_rate_all_none_severity() -> None:
    severities = [{"severity": "none"}] * 10
    assert anomaly_rate(severities) == pytest.approx(0.0)


def test_anomaly_rate_all_critical_simple() -> None:
    severities = [{"severity": "critical"}] * 5
    assert anomaly_rate(severities) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "n_anomalies,n_total,expected",
    [
        (1, 4, 0.25),
        (2, 4, 0.5),
        (3, 4, 0.75),
    ],
)
def test_anomaly_rate_partial(n_anomalies, n_total, expected) -> None:
    severities = [{"severity": "warning"}] * n_anomalies + [{"severity": "none"}] * (n_total - n_anomalies)
    assert anomaly_rate(severities) == pytest.approx(expected)


class TestEwmaSmooth:
    def test_single_value(self) -> None:
        from app.anomaly import ewma_smooth

        assert ewma_smooth([5.0]) == [5.0]

    def test_smoothing_reduces_spike(self) -> None:
        from app.anomaly import ewma_smooth

        values = [1.0, 1.0, 100.0, 1.0, 1.0]
        result = ewma_smooth(values, alpha=0.3)
        assert result[2] < 100.0

    def test_alpha_one_is_identity(self) -> None:
        from app.anomaly import ewma_smooth

        values = [1.0, 2.0, 3.0]
        result = ewma_smooth(values, alpha=1.0)
        assert result == pytest.approx([1.0, 2.0, 3.0])

    def test_raises_empty(self) -> None:
        import pytest

        from app.anomaly import ewma_smooth

        with pytest.raises(ValueError):
            ewma_smooth([])

    def test_raises_bad_alpha(self) -> None:
        import pytest

        from app.anomaly import ewma_smooth

        with pytest.raises(ValueError):
            ewma_smooth([1.0], alpha=0.0)


class TestConsecutiveAnomalyRuns:
    def test_single_run(self) -> None:
        from app.anomaly import consecutive_anomaly_runs

        flags = [False, True, True, True, False]
        assert consecutive_anomaly_runs(flags) == [(1, 3)]

    def test_no_anomalies(self) -> None:
        from app.anomaly import consecutive_anomaly_runs

        assert consecutive_anomaly_runs([False, False]) == []

    def test_trailing_run(self) -> None:
        from app.anomaly import consecutive_anomaly_runs

        flags = [False, True, True]
        assert consecutive_anomaly_runs(flags) == [(1, 2)]

    def test_multiple_runs(self) -> None:
        from app.anomaly import consecutive_anomaly_runs

        flags = [True, False, True]
        runs = consecutive_anomaly_runs(flags)
        assert len(runs) == 2

    def test_all_true(self) -> None:
        from app.anomaly import consecutive_anomaly_runs

        assert consecutive_anomaly_runs([True, True, True]) == [(0, 2)]


class TestFlagAnomalyRate:
    def test_empty(self) -> None:
        from app.anomaly import flag_anomaly_rate

        assert flag_anomaly_rate([]) == 0.0

    def test_none_flagged(self) -> None:
        from app.anomaly import flag_anomaly_rate

        assert flag_anomaly_rate([False, False, False]) == 0.0

    def test_all_flagged(self) -> None:
        from app.anomaly import flag_anomaly_rate

        assert flag_anomaly_rate([True, True]) == 1.0

    def test_half_flagged(self) -> None:
        from app.anomaly import flag_anomaly_rate

        assert flag_anomaly_rate([True, False]) == pytest.approx(0.5)


class TestAnomalyDensity:
    def test_no_anomalies(self) -> None:
        from app.anomaly import anomaly_density

        result = anomaly_density([0] * 10, window_size=5)
        assert all(v == 0.0 for v in result)

    def test_all_anomalies(self) -> None:
        from app.anomaly import anomaly_density

        result = anomaly_density([1] * 10, window_size=5)
        assert all(v == 1.0 for v in result)

    def test_empty_raises(self) -> None:
        from app.anomaly import anomaly_density

        with pytest.raises(ValueError, match="empty"):
            anomaly_density([], window_size=5)

    def test_window_zero_raises(self) -> None:
        from app.anomaly import anomaly_density

        with pytest.raises(ValueError, match="at least 1"):
            anomaly_density([0, 1, 0], window_size=0)

    def test_output_length(self) -> None:
        from app.anomaly import anomaly_density

        flags = [0, 1, 0, 1, 0, 0, 1]
        assert len(anomaly_density(flags, window_size=3)) == 7

    @pytest.mark.parametrize("window", [1, 3, 6])
    def test_values_in_range(self, window: int) -> None:
        from app.anomaly import anomaly_density

        flags = [0, 1, 0, 1, 0, 1, 0, 1]
        result = anomaly_density(flags, window_size=window)
        assert all(0.0 <= v <= 1.0 for v in result)


class TestAnomalyBurstScore:
    def test_no_anomalies_zero_score(self) -> None:
        from app.anomaly import anomaly_burst_score

        assert anomaly_burst_score([0] * 12, burst_window=4) == 0.0

    def test_all_anomalies_one_score(self) -> None:
        from app.anomaly import anomaly_burst_score

        assert anomaly_burst_score([1] * 8, burst_window=4) == 1.0

    def test_empty_raises(self) -> None:
        from app.anomaly import anomaly_burst_score

        with pytest.raises(ValueError, match="empty"):
            anomaly_burst_score([], burst_window=3)

    def test_burst_window_zero_raises(self) -> None:
        from app.anomaly import anomaly_burst_score

        with pytest.raises(ValueError, match="at least 1"):
            anomaly_burst_score([1, 0, 1], burst_window=0)

    def test_result_in_range(self) -> None:
        from app.anomaly import anomaly_burst_score

        score = anomaly_burst_score([0, 1, 1, 0, 0, 1], burst_window=3)
        assert 0.0 <= score <= 1.0


class TestPercentileAnomalyFlag:
    def test_value_within_range(self) -> None:
        from app.anomaly import percentile_anomaly_flag

        ref = list(range(1, 101))
        assert percentile_anomaly_flag(50.0, ref) is False

    def test_value_above_range(self) -> None:
        from app.anomaly import percentile_anomaly_flag

        ref = list(range(1, 101))
        assert percentile_anomaly_flag(200.0, ref) is True

    def test_value_below_range(self) -> None:
        from app.anomaly import percentile_anomaly_flag

        ref = list(range(1, 101))
        assert percentile_anomaly_flag(-5.0, ref) is True

    def test_empty_reference_raises(self) -> None:
        from app.anomaly import percentile_anomaly_flag

        with pytest.raises(ValueError, match="empty"):
            percentile_anomaly_flag(5.0, [])

    def test_invalid_percentiles_raises(self) -> None:
        from app.anomaly import percentile_anomaly_flag

        with pytest.raises(ValueError):
            percentile_anomaly_flag(5.0, [1.0, 2.0], lower_pct=90.0, upper_pct=10.0)


class TestConsecutiveNormalRuns:
    def test_all_normal(self) -> None:
        from app.anomaly import consecutive_normal_runs

        result = consecutive_normal_runs([0, 0, 0, 0])
        assert result == [(0, 3)]

    def test_all_anomalous(self) -> None:
        from app.anomaly import consecutive_normal_runs

        result = consecutive_normal_runs([1, 1, 1])
        assert result == []

    def test_empty_returns_empty(self) -> None:
        from app.anomaly import consecutive_normal_runs

        assert consecutive_normal_runs([]) == []

    def test_alternating(self) -> None:
        from app.anomaly import consecutive_normal_runs

        flags = [0, 1, 0, 1, 0]
        result = consecutive_normal_runs(flags)
        assert len(result) == 3

    def test_single_normal(self) -> None:
        from app.anomaly import consecutive_normal_runs

        assert consecutive_normal_runs([0]) == [(0, 0)]


def test_anomaly_summary_empty() -> None:
    assert anomaly_summary([]) == {"none": 0, "warning": 0, "critical": 0}


def test_anomaly_summary_counts() -> None:
    sevs = [
        {"severity": "none"},
        {"severity": "warning"},
        {"severity": "critical"},
        {"severity": "critical"},
    ]
    s = anomaly_summary(sevs)
    assert s["none"] == 1
    assert s["warning"] == 1
    assert s["critical"] == 2


def test_anomaly_summary_all_none() -> None:
    sevs = [{"severity": "none"}] * 5
    s = anomaly_summary(sevs)
    assert s["none"] == 5
    assert s["warning"] == 0
    assert s["critical"] == 0


def test_rolling_anomaly_flag_length() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 3.0, 2.0, 1.0]
    flags = rolling_anomaly_flag(values, window=5)
    assert len(flags) == len(values)


def test_rolling_anomaly_flag_empty_raises() -> None:
    with pytest.raises(ValueError):
        rolling_anomaly_flag([])


def test_rolling_anomaly_flag_small_window_raises() -> None:
    with pytest.raises(ValueError):
        rolling_anomaly_flag([1.0, 2.0], window=1)


def test_rolling_anomaly_flag_constant_no_flags() -> None:
    values = [5.0] * 20
    flags = rolling_anomaly_flag(values, window=5)
    assert not any(flags)


def test_rolling_anomaly_flag_outlier_flagged() -> None:
    values = [1.0] * 10 + [1000.0] + [1.0] * 10
    flags = rolling_anomaly_flag(values, window=5)
    assert flags[10] is True


@pytest.mark.parametrize("n", [5, 10, 20])
def test_rolling_anomaly_flag_length_param(n: int) -> None:
    values = list(range(n))
    flags = rolling_anomaly_flag(values, window=3)
    assert len(flags) == n


def test_zscore_flag_zero_std_no_flag() -> None:
    """Zero std should never flag (constant distribution)."""
    assert not zscore_flag(5.0, 5.0, 0.0, threshold=3.0)


def test_zscore_flag_negative_value_below_mean() -> None:
    assert zscore_flag(-10.0, 0.0, 2.0, threshold=3.0)


@pytest.mark.parametrize(
    "value,mean,std,threshold,expected",
    [
        (0.0, 0.0, 1.0, 3.0, False),
        (3.1, 0.0, 1.0, 3.0, True),
        (2.9, 0.0, 1.0, 3.0, False),
        (100.0, 0.0, 1.0, 2.0, True),
    ],
)
def test_zscore_flag_new_parametrized(value: float, mean: float, std: float, threshold: float, expected: bool) -> None:
    assert zscore_flag(value, mean, std, threshold) is expected


def test_rolling_anomaly_flag_all_same_window() -> None:
    values = [3.0] * 15
    flags = rolling_anomaly_flag(values, window=5)
    assert all(f is False for f in flags)


def test_rolling_anomaly_flag_spike_at_end() -> None:
    values = [1.0] * 15 + [999.0]
    flags = rolling_anomaly_flag(values, window=5)
    assert flags[-1] is True


class TestAnomalyFreeStreak:
    def test_all_normal_returns_full_length(self) -> None:
        from app.anomaly import anomaly_free_streak

        assert anomaly_free_streak([False, False, False]) == 3

    def test_ends_with_anomaly_returns_zero(self) -> None:
        from app.anomaly import anomaly_free_streak

        assert anomaly_free_streak([False, False, True]) == 0

    def test_mixed_trailing_normal(self) -> None:
        from app.anomaly import anomaly_free_streak

        assert anomaly_free_streak([True, False, False, False]) == 3

    def test_empty_list_returns_zero(self) -> None:
        from app.anomaly import anomaly_free_streak

        assert anomaly_free_streak([]) == 0

    def test_single_anomaly(self) -> None:
        from app.anomaly import anomaly_free_streak

        assert anomaly_free_streak([True]) == 0

    def test_single_normal(self) -> None:
        from app.anomaly import anomaly_free_streak

        assert anomaly_free_streak([False]) == 1


class TestAnomalyTransitionCount:
    def test_no_transitions_all_normal(self) -> None:
        from app.anomaly import anomaly_transition_count

        assert anomaly_transition_count([False, False, False]) == 0

    def test_single_transition(self) -> None:
        from app.anomaly import anomaly_transition_count

        assert anomaly_transition_count([False, True, True]) == 1

    def test_alternating_transitions(self) -> None:
        from app.anomaly import anomaly_transition_count

        assert anomaly_transition_count([False, True, False, True]) == 3

    def test_empty_returns_zero(self) -> None:
        from app.anomaly import anomaly_transition_count

        assert anomaly_transition_count([]) == 0

    def test_single_element_returns_zero(self) -> None:
        from app.anomaly import anomaly_transition_count

        assert anomaly_transition_count([False]) == 0


class TestAnomalyRateByHour:
    def test_basic(self) -> None:
        from app.anomaly import anomaly_rate_by_hour

        readings = [(0, True), (0, False), (1, True), (1, True)]
        result = anomaly_rate_by_hour(readings)
        assert result[0] == pytest.approx(0.5)
        assert result[1] == pytest.approx(1.0)

    def test_no_anomalies(self) -> None:
        from app.anomaly import anomaly_rate_by_hour

        readings = [(12, False), (12, False)]
        assert anomaly_rate_by_hour(readings) == {12: 0.0}

    def test_invalid_hour_raises(self) -> None:
        from app.anomaly import anomaly_rate_by_hour

        with pytest.raises(ValueError, match="0-23"):
            anomaly_rate_by_hour([(24, True)])

    @pytest.mark.parametrize("hour", [0, 11, 23])
    def test_boundary_hours(self, hour: int) -> None:
        from app.anomaly import anomaly_rate_by_hour

        result = anomaly_rate_by_hour([(hour, True)])
        assert hour in result


class TestSeverityWeightedScore:
    def test_default_weights(self) -> None:
        from app.anomaly import severity_weighted_score

        anomalies = [{"severity": "critical"}, {"severity": "warning"}]
        result = severity_weighted_score(anomalies)
        assert result == pytest.approx(4.5)

    def test_empty_list(self) -> None:
        from app.anomaly import severity_weighted_score

        assert severity_weighted_score([]) == 0.0

    def test_custom_weights(self) -> None:
        from app.anomaly import severity_weighted_score

        anomalies = [{"severity": "critical"}]
        result = severity_weighted_score(anomalies, severity_weights={"critical": 10.0})
        assert result == pytest.approx(10.0)

    def test_unknown_severity_zero(self) -> None:
        from app.anomaly import severity_weighted_score

        anomalies = [{"severity": "unknown"}]
        assert severity_weighted_score(anomalies) == 0.0


class TestBatchAnomalyFlag:
    def test_no_anomalies(self) -> None:
        from app.anomaly import batch_anomaly_flag

        result = batch_anomaly_flag([0.0, 1.0, -1.0], mean=0.0, std=1.0, threshold=3.0)
        assert result == [False, False, False]

    def test_anomaly_detected(self) -> None:
        from app.anomaly import batch_anomaly_flag

        result = batch_anomaly_flag([0.0, 100.0], mean=0.0, std=1.0, threshold=3.0)
        assert result == [False, True]

    def test_zero_std_raises(self) -> None:
        from app.anomaly import batch_anomaly_flag

        with pytest.raises(ValueError, match="std"):
            batch_anomaly_flag([1.0, 2.0], mean=0.0, std=0.0)

    def test_zero_threshold_raises(self) -> None:
        from app.anomaly import batch_anomaly_flag

        with pytest.raises(ValueError, match="threshold"):
            batch_anomaly_flag([1.0], mean=0.0, std=1.0, threshold=0.0)

    def test_output_length(self) -> None:
        from app.anomaly import batch_anomaly_flag

        values = [float(i) for i in range(10)]
        result = batch_anomaly_flag(values, mean=4.5, std=3.0)
        assert len(result) == 10

    @pytest.mark.parametrize("threshold", [1.0, 2.0, 3.0])
    def test_threshold_sensitivity(self, threshold: float) -> None:
        from app.anomaly import batch_anomaly_flag

        values = [0.0, 5.0, -5.0, 0.5]
        result = batch_anomaly_flag(values, mean=0.0, std=1.0, threshold=threshold)
        assert isinstance(result, list)
        assert len(result) == len(values)


# ---------------------------------------------------------------------------
# Tests for anomaly_score_ema, mean_time_between_anomalies, anomaly_peak_ratio
# ---------------------------------------------------------------------------


class TestAnomalyScoreEma:
    def test_all_false_returns_zeros(self) -> None:
        from app.anomaly import anomaly_score_ema

        result = anomaly_score_ema([False, False, False], alpha=0.3)
        assert result == [0.0, 0.0, 0.0]

    def test_all_true_converges(self) -> None:
        from app.anomaly import anomaly_score_ema

        result = anomaly_score_ema([True] * 10, alpha=0.5)
        assert result[-1] > 0.9

    def test_length_matches_input(self) -> None:
        from app.anomaly import anomaly_score_ema

        flags = [True, False, True, False, True]
        assert len(anomaly_score_ema(flags)) == len(flags)

    def test_invalid_alpha_raises(self) -> None:
        import pytest

        from app.anomaly import anomaly_score_ema

        with pytest.raises(ValueError):
            anomaly_score_ema([True], alpha=0.0)

    def test_alpha_one_is_identity(self) -> None:
        from app.anomaly import anomaly_score_ema

        flags = [True, False, True]
        result = anomaly_score_ema(flags, alpha=1.0)
        assert result == [1.0, 0.0, 1.0]


class TestMeanTimeBetweenAnomalies:
    def test_no_anomalies(self) -> None:
        from app.anomaly import mean_time_between_anomalies

        assert mean_time_between_anomalies([False, False, False]) == float("inf")

    def test_one_anomaly(self) -> None:
        from app.anomaly import mean_time_between_anomalies

        assert mean_time_between_anomalies([False, True, False]) == float("inf")

    def test_two_anomalies(self) -> None:
        from app.anomaly import mean_time_between_anomalies

        flags = [True, False, False, True]
        assert mean_time_between_anomalies(flags) == 3.0

    def test_regular_spacing(self) -> None:
        from app.anomaly import mean_time_between_anomalies

        flags = [True, False, True, False, True]
        assert mean_time_between_anomalies(flags) == 2.0


class TestAnomalyPeakRatio:
    def test_empty_groups_return_zero(self) -> None:
        from app.anomaly import anomaly_peak_ratio

        assert anomaly_peak_ratio([1.0, 2.0], [False, False]) == 0.0

    def test_all_anomalous_returns_zero(self) -> None:
        from app.anomaly import anomaly_peak_ratio

        assert anomaly_peak_ratio([1.0, 2.0], [True, True]) == 0.0

    def test_length_mismatch_raises(self) -> None:
        import pytest

        from app.anomaly import anomaly_peak_ratio

        with pytest.raises(ValueError):
            anomaly_peak_ratio([1.0, 2.0], [True])

    def test_ratio_greater_one_for_high_anomalies(self) -> None:
        from app.anomaly import anomaly_peak_ratio

        values = [1.0, 1.0, 100.0]
        flags = [False, False, True]
        assert anomaly_peak_ratio(values, flags) > 1.0


@pytest.mark.parametrize(
    "value,mean,std,threshold,expected",
    [
        (10.0, 10.0, 1.0, 3.0, False),
        (100.0, 10.0, 1.0, 3.0, True),
        (12.0, 10.0, 1.0, 1.5, True),
        (10.5, 10.0, 1.0, 3.0, False),
    ],
)
def test_zscore_flag_various_cases(value: float, mean: float, std: float, threshold: float, expected: bool) -> None:
    from app.anomaly import zscore_flag

    assert zscore_flag(value, mean, std, threshold) == expected


@pytest.mark.parametrize(
    "flags,expected_rate",
    [
        ([False, False, False, False], 0.0),
        ([True, True, True, True], 1.0),
        ([True, False, True, False], 0.5),
    ],
)
def test_flag_anomaly_rate_parametrized(flags: list, expected_rate: float) -> None:
    from app.anomaly import flag_anomaly_rate

    assert flag_anomaly_rate(flags) == pytest.approx(expected_rate, abs=0.01)


@pytest.mark.parametrize("n_anomalies", [0, 1, 5, 10])
def test_anomaly_free_streak_all_normal(n_anomalies: int) -> None:
    from app.anomaly import anomaly_free_streak

    flags = [False] * 10
    result = anomaly_free_streak(flags)
    assert result == 10


@pytest.mark.parametrize(
    "severities,expected_total",
    [
        ([], 0),
        ([{"level": "high"}, {"level": "low"}], 2),
    ],
)
def test_anomaly_summary_total_count(severities: list, expected_total: int) -> None:
    from app.anomaly import anomaly_summary

    result = anomaly_summary(severities)
    total = sum(result.values())
    assert total == expected_total


class TestAnomalyRate:
    def test_no_anomalies_returns_zero(self) -> None:
        from app.anomaly import anomaly_rate

        sevs = [{"severity": "none"}, {"severity": "none"}]
        assert anomaly_rate(sevs) == 0.0

    def test_all_anomalies_returns_one(self) -> None:
        from app.anomaly import anomaly_rate

        sevs = [{"severity": "warning"}, {"severity": "critical"}, {"severity": "warning"}]
        assert anomaly_rate(sevs) == 1.0

    def test_empty_returns_zero(self) -> None:
        from app.anomaly import anomaly_rate

        assert anomaly_rate([]) == 0.0

    @pytest.mark.parametrize(
        "sevs,expected",
        [
            ([{"severity": "none"}, {"severity": "none"}, {"severity": "critical"}], pytest.approx(1 / 3, rel=1e-4)),
            ([{"severity": "warning"}, {"severity": "critical"}, {"severity": "none"}], pytest.approx(2 / 3, rel=1e-4)),
        ],
    )
    def test_partial_anomaly_rate(self, sevs: list, expected: object) -> None:
        from app.anomaly import anomaly_rate

        assert anomaly_rate(sevs) == expected


class TestRollingAnomalyFlag:
    def test_output_same_length(self) -> None:
        from app.anomaly import rolling_anomaly_flag

        values = [1.0] * 20
        result = rolling_anomaly_flag(values, window=5)
        assert len(result) == 20

    def test_first_window_all_false(self) -> None:
        from app.anomaly import rolling_anomaly_flag

        result = rolling_anomaly_flag([1.0] * 15, window=10)
        assert all(not f for f in result[:10])

    def test_returns_list_of_bool(self) -> None:
        from app.anomaly import rolling_anomaly_flag

        result = rolling_anomaly_flag([1.0, 2.0, 3.0, 4.0, 5.0])
        assert all(isinstance(f, bool) for f in result)


class TestAnomalyPersistenceScore:
    def test_all_anomalies(self) -> None:
        from app.anomaly import anomaly_persistence_score
        assert anomaly_persistence_score([True, True, True]) == pytest.approx(1.0)

    def test_no_anomalies(self) -> None:
        from app.anomaly import anomaly_persistence_score
        assert anomaly_persistence_score([False, False, False]) == pytest.approx(0.0)

    def test_partial(self) -> None:
        from app.anomaly import anomaly_persistence_score
        assert anomaly_persistence_score([True, False, True]) == pytest.approx(2 / 3)

    def test_empty_raises(self) -> None:
        from app.anomaly import anomaly_persistence_score
        with pytest.raises(ValueError):
            anomaly_persistence_score([])


class TestFirstAnomalyIndex:
    def test_found(self) -> None:
        from app.anomaly import first_anomaly_index
        assert first_anomaly_index([False, False, True, False]) == 2

    def test_not_found(self) -> None:
        from app.anomaly import first_anomaly_index
        assert first_anomaly_index([False, False]) == -1

    def test_first_element(self) -> None:
        from app.anomaly import first_anomaly_index
        assert first_anomaly_index([True, False]) == 0


class TestInterAnomalyGap:
    def test_regular_gaps(self) -> None:
        from app.anomaly import inter_anomaly_gap
        assert inter_anomaly_gap([True, False, False, True, False, False, True]) == pytest.approx(3.0)

    def test_no_anomaly(self) -> None:
        from app.anomaly import inter_anomaly_gap
        assert inter_anomaly_gap([False, False, False]) == float("inf")

    def test_one_anomaly(self) -> None:
        from app.anomaly import inter_anomaly_gap
        assert inter_anomaly_gap([False, True, False]) == float("inf")

    def test_empty_raises(self) -> None:
        from app.anomaly import inter_anomaly_gap
        with pytest.raises(ValueError):
            inter_anomaly_gap([])
