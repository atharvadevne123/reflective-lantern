"""Energy reporting tests."""

from __future__ import annotations

import pytest

from app.reporting import estimate_savings, peak_demand_report


def test_savings_positive() -> None:
    result = estimate_savings([8.0] * 10, [10.0] * 10)
    assert result["total_saved_kwh"] == pytest.approx(20.0)
    assert result["savings_pct"] == pytest.approx(20.0)


def test_savings_negative() -> None:
    result = estimate_savings([12.0] * 5, [10.0] * 5)
    assert result["total_saved_kwh"] < 0


def test_peak_demand() -> None:
    hourly = [5.0] * 24
    hourly[14] = 25.0
    r = peak_demand_report(hourly)
    assert r["peak_hour"] == 14
    assert r["peak_kwh"] == pytest.approx(25.0)


def test_demand_factor() -> None:
    hourly = [10.0] * 24
    hourly[0] = 30.0
    r = peak_demand_report(hourly)
    assert r["demand_factor"] > 1.0


def test_savings_zero_baseline() -> None:
    result = estimate_savings([0.0], [0.0])
    assert result["savings_pct"] == 0.0


@pytest.mark.parametrize("tariff,expected_cost", [(0.10, 10.0), (0.20, 20.0), (0.15, 15.0)])
def test_savings_cost_with_tariff(tariff: float, expected_cost: float) -> None:
    result = estimate_savings([0.0] * 10, [10.0] * 10, tariff_per_kwh=tariff)
    assert result["total_saved_cost"] == pytest.approx(expected_cost)


def test_savings_returns_all_keys() -> None:
    result = estimate_savings([5.0], [10.0])
    assert "total_saved_kwh" in result
    assert "total_saved_cost" in result
    assert "savings_pct" in result


def test_peak_demand_off_peak_mean() -> None:
    hourly = [10.0] * 24
    r = peak_demand_report(hourly)
    assert r["off_peak_mean"] == pytest.approx(10.0)
    assert r["demand_factor"] == pytest.approx(1.0)


def test_estimate_savings_single_value() -> None:
    result = estimate_savings([5.0], [8.0])
    assert result["total_saved_kwh"] == pytest.approx(3.0)
    assert result["savings_pct"] == pytest.approx(37.5)


def test_estimate_savings_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        estimate_savings([1.0, 2.0], [1.0])


def test_peak_demand_empty_raises() -> None:
    with pytest.raises(ValueError):
        peak_demand_report([])


@pytest.mark.parametrize("n_hours", [1, 12, 24, 48])
def test_peak_demand_various_lengths(n_hours: int) -> None:
    hourly = list(range(n_hours, 0, -1))
    r = peak_demand_report([float(v) for v in hourly])
    assert r["peak_hour"] == 0  # largest is first element


@pytest.mark.parametrize(
    "actual,baseline,expected_grade",
    [
        (8.0, 10.0, "A+"),  # 20% reduction -> A+
        (5.0, 10.0, "A+"),  # 50% reduction
        (9.5, 10.0, "A-"),  # 5% reduction
        (10.0, 10.0, "B"),  # 0% reduction
        (10.6, 10.0, "D"),  # slight increase -> D
        (12.0, 10.0, "F"),  # 20% increase
    ],
)
def test_energy_efficiency_grade_parametrized(actual, baseline, expected_grade):
    from app.reporting import energy_efficiency_grade

    assert energy_efficiency_grade(actual, baseline) == expected_grade


def test_energy_efficiency_grade_zero_baseline():
    from app.reporting import energy_efficiency_grade

    assert energy_efficiency_grade(10.0, 0.0) == "F"


def test_monthly_consumption_summary_basic():
    from app.reporting import monthly_consumption_summary

    daily = [10.0 + i * 0.1 for i in range(30)]
    result = monthly_consumption_summary(daily)
    assert "total_kwh" in result
    assert "mean_kwh" in result
    assert "max_kwh" in result
    assert "min_kwh" in result
    assert "std_kwh" in result
    assert "estimated_cost" in result
    assert result["days"] == 30


def test_monthly_consumption_summary_empty():
    from app.reporting import monthly_consumption_summary

    with pytest.raises(ValueError):
        monthly_consumption_summary([])


def test_monthly_consumption_summary_cost():
    from app.reporting import monthly_consumption_summary

    result = monthly_consumption_summary([100.0], tariff_per_kwh=0.10)
    assert result["estimated_cost"] == pytest.approx(10.0)


def test_consumption_trend_rising():
    from app.reporting import consumption_trend

    data = [10.0 + i * 2 for i in range(20)]
    assert consumption_trend(data) == "rising"


def test_consumption_trend_falling():
    from app.reporting import consumption_trend

    data = [100.0 - i * 2 for i in range(20)]
    assert consumption_trend(data) == "falling"


def test_consumption_trend_stable():
    from app.reporting import consumption_trend

    data = [50.0] * 30
    assert consumption_trend(data) == "stable"


def test_consumption_trend_single_value():
    from app.reporting import consumption_trend

    assert consumption_trend([42.0]) == "stable"


def test_consumption_trend_empty():
    from app.reporting import consumption_trend

    assert consumption_trend([]) == "stable"


@pytest.mark.parametrize(
    "data,expected",
    [
        ([1.0, 2.0, 3.0, 4.0, 5.0], "rising"),
        ([5.0, 4.0, 3.0, 2.0, 1.0], "falling"),
        ([3.0, 3.0, 3.0, 3.0, 3.0], "stable"),
    ],
)
def test_consumption_trend_parametrized(data, expected):
    from app.reporting import consumption_trend

    assert consumption_trend(data) == expected


def test_seasonal_efficiency_score_keys():
    from app.reporting import seasonal_efficiency_score

    actual = [8.0] * 40
    baseline = [10.0] * 40
    result = seasonal_efficiency_score(actual, baseline)
    assert "q1_savings_pct" in result
    assert "q2_savings_pct" in result
    assert "q3_savings_pct" in result
    assert "q4_savings_pct" in result
    assert "overall_score" in result


def test_seasonal_efficiency_score_positive_savings():
    from app.reporting import seasonal_efficiency_score

    actual = [5.0] * 40
    baseline = [10.0] * 40
    result = seasonal_efficiency_score(actual, baseline)
    assert result["overall_score"] == pytest.approx(50.0, rel=1e-2)


def test_seasonal_efficiency_score_length_mismatch_raises():
    from app.reporting import seasonal_efficiency_score

    with pytest.raises(ValueError):
        seasonal_efficiency_score([1.0, 2.0], [1.0, 2.0, 3.0])


def test_seasonal_efficiency_score_empty_raises():
    from app.reporting import seasonal_efficiency_score

    with pytest.raises(ValueError):
        seasonal_efficiency_score([], [])


def test_seasonal_efficiency_score_custom_weights():
    from app.reporting import seasonal_efficiency_score

    actual = [8.0] * 40
    baseline = [10.0] * 40
    weights = {"q1": 0.5, "q2": 0.2, "q3": 0.2, "q4": 0.1}
    result = seasonal_efficiency_score(actual, baseline, season_weights=weights)
    assert "overall_score" in result


@pytest.mark.parametrize("n", [4, 12, 24, 48, 100])
def test_seasonal_efficiency_score_various_lengths(n):
    from app.reporting import seasonal_efficiency_score

    actual = [1.0] * n
    baseline = [2.0] * n
    result = seasonal_efficiency_score(actual, baseline)
    assert result["overall_score"] == pytest.approx(50.0, rel=1e-2)


def test_peak_demand_by_period_basic():
    from app.reporting import peak_demand_by_period

    data = [1.0] * 24
    data[8] = 10.0  # period starting at hour 8 (period_start=8 for period_hours=4)
    result = peak_demand_by_period(data, period_hours=4)
    peak_periods = [p for p in result if p["is_peak"]]
    assert len(peak_periods) == 1


def test_peak_demand_by_period_period_count():
    from app.reporting import peak_demand_by_period

    data = [1.0] * 24
    result = peak_demand_by_period(data, period_hours=4)
    assert len(result) == 6


def test_peak_demand_by_period_empty_raises():
    from app.reporting import peak_demand_by_period

    with pytest.raises(ValueError):
        peak_demand_by_period([])


def test_peak_demand_by_period_bad_period_raises():
    from app.reporting import peak_demand_by_period

    with pytest.raises(ValueError):
        peak_demand_by_period([1.0] * 24, period_hours=0)


def test_peak_demand_by_period_keys():
    from app.reporting import peak_demand_by_period

    result = peak_demand_by_period([5.0] * 8, period_hours=4)
    for p in result:
        assert set(p.keys()) >= {"period_start", "total_kwh", "is_peak"}


@pytest.mark.parametrize("period_hours,n_hours,expected_count", [
    (4, 24, 6),
    (6, 24, 4),
    (8, 24, 3),
    (1, 10, 10),
])
def test_peak_demand_by_period_counts(period_hours, n_hours, expected_count):
    from app.reporting import peak_demand_by_period

    result = peak_demand_by_period([1.0] * n_hours, period_hours=period_hours)
    assert len(result) == expected_count


def test_consumption_efficiency_ratio_under_target():
    from app.reporting import consumption_efficiency_ratio

    assert consumption_efficiency_ratio(80.0, 100.0) == pytest.approx(0.8)


def test_consumption_efficiency_ratio_over_target():
    from app.reporting import consumption_efficiency_ratio

    assert consumption_efficiency_ratio(120.0, 100.0) == pytest.approx(1.2)


def test_consumption_efficiency_ratio_zero_target():
    from app.reporting import consumption_efficiency_ratio

    assert consumption_efficiency_ratio(50.0, 0.0) == 0.0


def test_consumption_efficiency_ratio_equal():
    from app.reporting import consumption_efficiency_ratio

    assert consumption_efficiency_ratio(100.0, 100.0) == pytest.approx(1.0)


def test_daily_average_consumption_empty():
    from app.reporting import daily_average_consumption

    assert daily_average_consumption([]) == 0.0


def test_daily_average_consumption_one_day():
    from app.reporting import daily_average_consumption

    data = [1.0] * 24
    assert daily_average_consumption(data) == pytest.approx(24.0)


def test_daily_average_consumption_two_days():
    from app.reporting import daily_average_consumption

    data = [2.0] * 48
    assert daily_average_consumption(data) == pytest.approx(48.0)


def test_daily_average_consumption_partial_day():
    from app.reporting import daily_average_consumption

    data = [1.0] * 12
    result = daily_average_consumption(data)
    assert result > 0.0


@pytest.mark.parametrize("actual,target,expected", [
    (50.0, 100.0, 0.5),
    (100.0, 100.0, 1.0),
    (150.0, 100.0, 1.5),
])
def test_consumption_efficiency_ratio_parametrized(actual, target, expected):
    from app.reporting import consumption_efficiency_ratio

    assert consumption_efficiency_ratio(actual, target) == pytest.approx(expected)


def test_seasonal_efficiency_score_basic():
    from app.reporting import seasonal_efficiency_score
    actual = [8.0] * 40
    baseline = [10.0] * 40
    result = seasonal_efficiency_score(actual, baseline)
    assert "q1_savings_pct" in result
    assert "overall_score" in result
    assert result["overall_score"] > 0


def test_seasonal_efficiency_score_mismatched_raises():
    from app.reporting import seasonal_efficiency_score
    with pytest.raises(ValueError):
        seasonal_efficiency_score([1.0, 2.0], [3.0])


def test_seasonal_efficiency_score_empty_raises():
    from app.reporting import seasonal_efficiency_score
    with pytest.raises(ValueError):
        seasonal_efficiency_score([], [])


def test_consumption_trend_rising():
    from app.reporting import consumption_trend
    data = [float(i) for i in range(10, 40)]
    assert consumption_trend(data) == "rising"


def test_consumption_trend_falling():
    from app.reporting import consumption_trend
    data = [float(i) for i in range(30, 0, -1)]
    assert consumption_trend(data) == "falling"


def test_consumption_trend_stable():
    from app.reporting import consumption_trend
    data = [10.0] * 20
    assert consumption_trend(data) == "stable"


def test_consumption_trend_single_value():
    from app.reporting import consumption_trend
    assert consumption_trend([5.0]) == "stable"


def test_peak_demand_by_period_basic():
    from app.reporting import peak_demand_by_period
    hourly = [1.0] * 20 + [10.0] * 4
    result = peak_demand_by_period(hourly, period_hours=4)
    assert any(p["is_peak"] for p in result)


def test_peak_demand_by_period_empty_raises():
    from app.reporting import peak_demand_by_period
    with pytest.raises(ValueError):
        peak_demand_by_period([])


def test_peak_demand_by_period_invalid_period_raises():
    from app.reporting import peak_demand_by_period
    with pytest.raises(ValueError):
        peak_demand_by_period([1.0] * 10, period_hours=0)


@pytest.mark.parametrize("period_hours", [2, 4, 6])
def test_peak_demand_by_period_parametrized(period_hours):
    from app.reporting import peak_demand_by_period
    hourly = [float(i) for i in range(24)]
    result = peak_demand_by_period(hourly, period_hours=period_hours)
    assert len(result) >= 1
    assert any(p["is_peak"] for p in result)
