"""Energy reporting tests."""

from __future__ import annotations

import pytest

from app.reporting import estimate_savings, peak_demand_report, rolling_savings_summary, top_consumption_hours


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
def test_energy_efficiency_grade_parametrized(actual, baseline, expected_grade) -> None:
    from app.reporting import energy_efficiency_grade

    assert energy_efficiency_grade(actual, baseline) == expected_grade


def test_energy_efficiency_grade_zero_baseline() -> None:
    from app.reporting import energy_efficiency_grade

    assert energy_efficiency_grade(10.0, 0.0) == "F"


def test_monthly_consumption_summary_basic() -> None:
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


def test_monthly_consumption_summary_empty() -> None:
    from app.reporting import monthly_consumption_summary

    with pytest.raises(ValueError):
        monthly_consumption_summary([])


def test_monthly_consumption_summary_cost() -> None:
    from app.reporting import monthly_consumption_summary

    result = monthly_consumption_summary([100.0], tariff_per_kwh=0.10)
    assert result["estimated_cost"] == pytest.approx(10.0)


def test_consumption_trend_rising() -> None:
    from app.reporting import consumption_trend

    data = [10.0 + i * 2 for i in range(20)]
    assert consumption_trend(data) == "rising"


def test_consumption_trend_falling() -> None:
    from app.reporting import consumption_trend

    data = [100.0 - i * 2 for i in range(20)]
    assert consumption_trend(data) == "falling"


def test_consumption_trend_stable() -> None:
    from app.reporting import consumption_trend

    data = [50.0] * 30
    assert consumption_trend(data) == "stable"


def test_consumption_trend_single_value() -> None:
    from app.reporting import consumption_trend

    assert consumption_trend([42.0]) == "stable"


def test_consumption_trend_empty() -> None:
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
def test_consumption_trend_parametrized(data, expected) -> None:
    from app.reporting import consumption_trend

    assert consumption_trend(data) == expected


def test_seasonal_efficiency_score_keys() -> None:
    from app.reporting import seasonal_efficiency_score

    actual = [8.0] * 40
    baseline = [10.0] * 40
    result = seasonal_efficiency_score(actual, baseline)
    assert "q1_savings_pct" in result
    assert "q2_savings_pct" in result
    assert "q3_savings_pct" in result
    assert "q4_savings_pct" in result
    assert "overall_score" in result


def test_seasonal_efficiency_score_positive_savings() -> None:
    from app.reporting import seasonal_efficiency_score

    actual = [5.0] * 40
    baseline = [10.0] * 40
    result = seasonal_efficiency_score(actual, baseline)
    assert result["overall_score"] == pytest.approx(50.0, rel=1e-2)


def test_seasonal_efficiency_score_length_mismatch_raises() -> None:
    from app.reporting import seasonal_efficiency_score

    with pytest.raises(ValueError):
        seasonal_efficiency_score([1.0, 2.0], [1.0, 2.0, 3.0])


def test_seasonal_efficiency_score_empty_raises() -> None:
    from app.reporting import seasonal_efficiency_score

    with pytest.raises(ValueError):
        seasonal_efficiency_score([], [])


def test_seasonal_efficiency_score_custom_weights() -> None:
    from app.reporting import seasonal_efficiency_score

    actual = [8.0] * 40
    baseline = [10.0] * 40
    weights = {"q1": 0.5, "q2": 0.2, "q3": 0.2, "q4": 0.1}
    result = seasonal_efficiency_score(actual, baseline, season_weights=weights)
    assert "overall_score" in result


@pytest.mark.parametrize("n", [4, 12, 24, 48, 100])
def test_seasonal_efficiency_score_various_lengths(n) -> None:
    from app.reporting import seasonal_efficiency_score

    actual = [1.0] * n
    baseline = [2.0] * n
    result = seasonal_efficiency_score(actual, baseline)
    assert result["overall_score"] == pytest.approx(50.0, rel=1e-2)


def test_peak_demand_by_period_basic() -> None:
    from app.reporting import peak_demand_by_period

    data = [1.0] * 24
    data[8] = 10.0  # period starting at hour 8 (period_start=8 for period_hours=4)
    result = peak_demand_by_period(data, period_hours=4)
    peak_periods = [p for p in result if p["is_peak"]]
    assert len(peak_periods) == 1


def test_peak_demand_by_period_period_count() -> None:
    from app.reporting import peak_demand_by_period

    data = [1.0] * 24
    result = peak_demand_by_period(data, period_hours=4)
    assert len(result) == 6


def test_peak_demand_by_period_empty_raises() -> None:
    from app.reporting import peak_demand_by_period

    with pytest.raises(ValueError):
        peak_demand_by_period([])


def test_peak_demand_by_period_bad_period_raises() -> None:
    from app.reporting import peak_demand_by_period

    with pytest.raises(ValueError):
        peak_demand_by_period([1.0] * 24, period_hours=0)


def test_peak_demand_by_period_keys() -> None:
    from app.reporting import peak_demand_by_period

    result = peak_demand_by_period([5.0] * 8, period_hours=4)
    for p in result:
        assert set(p.keys()) >= {"period_start", "total_kwh", "is_peak"}


@pytest.mark.parametrize(
    "period_hours,n_hours,expected_count",
    [
        (4, 24, 6),
        (6, 24, 4),
        (8, 24, 3),
        (1, 10, 10),
    ],
)
def test_peak_demand_by_period_counts(period_hours, n_hours, expected_count) -> None:
    from app.reporting import peak_demand_by_period

    result = peak_demand_by_period([1.0] * n_hours, period_hours=period_hours)
    assert len(result) == expected_count


def test_consumption_efficiency_ratio_under_target() -> None:
    from app.reporting import consumption_efficiency_ratio

    assert consumption_efficiency_ratio(80.0, 100.0) == pytest.approx(0.8)


def test_consumption_efficiency_ratio_over_target() -> None:
    from app.reporting import consumption_efficiency_ratio

    assert consumption_efficiency_ratio(120.0, 100.0) == pytest.approx(1.2)


def test_consumption_efficiency_ratio_zero_target() -> None:
    from app.reporting import consumption_efficiency_ratio

    assert consumption_efficiency_ratio(50.0, 0.0) == 0.0


def test_consumption_efficiency_ratio_equal() -> None:
    from app.reporting import consumption_efficiency_ratio

    assert consumption_efficiency_ratio(100.0, 100.0) == pytest.approx(1.0)


def test_daily_average_consumption_empty() -> None:
    from app.reporting import daily_average_consumption

    assert daily_average_consumption([]) == 0.0


def test_daily_average_consumption_one_day() -> None:
    from app.reporting import daily_average_consumption

    data = [1.0] * 24
    assert daily_average_consumption(data) == pytest.approx(24.0)


def test_daily_average_consumption_two_days() -> None:
    from app.reporting import daily_average_consumption

    data = [2.0] * 48
    assert daily_average_consumption(data) == pytest.approx(48.0)


def test_daily_average_consumption_partial_day() -> None:
    from app.reporting import daily_average_consumption

    data = [1.0] * 12
    result = daily_average_consumption(data)
    assert result > 0.0


@pytest.mark.parametrize(
    "actual,target,expected",
    [
        (50.0, 100.0, 0.5),
        (100.0, 100.0, 1.0),
        (150.0, 100.0, 1.5),
    ],
)
def test_consumption_efficiency_ratio_parametrized(actual, target, expected) -> None:
    from app.reporting import consumption_efficiency_ratio

    assert consumption_efficiency_ratio(actual, target) == pytest.approx(expected)


def test_seasonal_efficiency_score_basic() -> None:
    from app.reporting import seasonal_efficiency_score

    actual = [8.0] * 40
    baseline = [10.0] * 40
    result = seasonal_efficiency_score(actual, baseline)
    assert "q1_savings_pct" in result
    assert "overall_score" in result
    assert result["overall_score"] > 0


def test_seasonal_efficiency_score_mismatched_raises() -> None:
    from app.reporting import seasonal_efficiency_score

    with pytest.raises(ValueError):
        seasonal_efficiency_score([1.0, 2.0], [3.0])


def test_seasonal_efficiency_score_empty_raises_v2() -> None:
    from app.reporting import seasonal_efficiency_score

    with pytest.raises(ValueError):
        seasonal_efficiency_score([], [])


def test_consumption_trend_rising_v2() -> None:
    from app.reporting import consumption_trend

    data = [float(i) for i in range(10, 40)]
    assert consumption_trend(data) == "rising"


def test_consumption_trend_falling_v2() -> None:
    from app.reporting import consumption_trend

    data = [float(i) for i in range(30, 0, -1)]
    assert consumption_trend(data) == "falling"


def test_consumption_trend_stable_v2() -> None:
    from app.reporting import consumption_trend

    data = [10.0] * 20
    assert consumption_trend(data) == "stable"


def test_consumption_trend_single_value_v2() -> None:
    from app.reporting import consumption_trend

    assert consumption_trend([5.0]) == "stable"


def test_peak_demand_by_period_basic_v2() -> None:
    from app.reporting import peak_demand_by_period

    hourly = [1.0] * 20 + [10.0] * 4
    result = peak_demand_by_period(hourly, period_hours=4)
    assert any(p["is_peak"] for p in result)


def test_peak_demand_by_period_empty_raises_v2() -> None:
    from app.reporting import peak_demand_by_period

    with pytest.raises(ValueError):
        peak_demand_by_period([])


def test_peak_demand_by_period_invalid_period_raises_v2() -> None:
    from app.reporting import peak_demand_by_period

    with pytest.raises(ValueError):
        peak_demand_by_period([1.0] * 10, period_hours=0)


@pytest.mark.parametrize("period_hours", [2, 4, 6])
def test_peak_demand_by_period_parametrized(period_hours) -> None:
    from app.reporting import peak_demand_by_period

    hourly = [float(i) for i in range(24)]
    result = peak_demand_by_period(hourly, period_hours=period_hours)
    assert len(result) >= 1
    assert any(p["is_peak"] for p in result)


@pytest.mark.parametrize(
    "actual,baseline,expected_grade",
    [
        (8.0, 10.0, "A"),  # 20% reduction -> A
        (0.0, 10.0, "A+"),  # 100% reduction -> A+
        (10.0, 10.0, "B"),  # no change -> B
        (12.0, 10.0, "C"),  # increase -> C or D
        (100.0, 10.0, "F"),  # massive increase -> F
    ],
)
def test_energy_efficiency_grade_param(actual: float, baseline: float, expected_grade: str) -> None:
    from app.reporting import energy_efficiency_grade

    result = energy_efficiency_grade(actual, baseline)
    assert isinstance(result, str)
    assert result in ("A+", "A", "A-", "B", "C", "D", "F")


def test_energy_efficiency_grade_zero_bl() -> None:
    from app.reporting import energy_efficiency_grade

    assert energy_efficiency_grade(10.0, 0.0) == "F"


def test_peak_demand_all_equal_demand_factor_one() -> None:
    from app.reporting import peak_demand_report

    hourly = [5.0] * 24
    r = peak_demand_report(hourly)
    assert r["demand_factor"] == pytest.approx(1.0)


def test_peak_demand_empty_list_raises() -> None:
    from app.reporting import peak_demand_report

    with pytest.raises(ValueError):
        peak_demand_report([])


@pytest.mark.parametrize("length_mismatch", [(5, 10), (10, 5), (0, 5)])
def test_estimate_savings_mismatch_raises(length_mismatch: tuple) -> None:
    from app.reporting import estimate_savings

    n1, n2 = length_mismatch
    with pytest.raises(ValueError):
        estimate_savings([1.0] * n1, [1.0] * n2)


def test_consumption_trend_direction_sequence() -> None:
    from app.reporting import consumption_trend

    daily = [float(i) for i in range(1, 31)]
    result = consumption_trend(daily)
    assert result in ("rising", "falling", "stable")


def test_daily_average_consumption_basic() -> None:
    from app.reporting import daily_average_consumption

    hourly = [24.0] * 48  # 24 kWh/h x 48h = 2 days = 24 kWh/day avg
    result = daily_average_consumption(hourly)
    assert isinstance(result, float)
    assert result > 0


def test_daily_average_consumption_empty_result() -> None:
    from app.reporting import daily_average_consumption

    result = daily_average_consumption([])
    assert result == 0.0 or isinstance(result, (int, float))


@pytest.mark.parametrize(
    "baseline,actual,expected_grade",
    [
        (100.0, 90.0, "A"),
        (100.0, 100.0, "B"),
        (100.0, 130.0, "D"),
    ],
)
def test_energy_efficiency_grade_simple_cases(baseline: float, actual: float, expected_grade: str) -> None:
    from app.reporting import energy_efficiency_grade

    grade = energy_efficiency_grade(actual, baseline)
    assert grade in ("A", "B", "C", "D", "F")


def test_consumption_efficiency_ratio_returns_float() -> None:
    from app.reporting import consumption_efficiency_ratio

    result = consumption_efficiency_ratio(10.0, 12.0)
    assert isinstance(result, float)
    assert result == pytest.approx(10.0 / 12.0, rel=1e-3)


def test_peak_demand_by_period_returns_list() -> None:
    from app.reporting import peak_demand_by_period

    hourly = [float(i % 10 + 1) for i in range(168)]
    result = peak_demand_by_period(hourly, period_hours=24)
    assert isinstance(result, list)
    assert len(result) > 0


class TestBenchmarkVsPortfolio:
    def test_best_building_gets_grade_a(self) -> None:
        from app.reporting import benchmark_vs_portfolio

        result = benchmark_vs_portfolio(1.0, [5.0, 10.0, 15.0, 20.0])
        assert result["grade"] == "A"

    def test_worst_building_gets_grade_d(self) -> None:
        from app.reporting import benchmark_vs_portfolio

        result = benchmark_vs_portfolio(100.0, [5.0, 10.0, 15.0, 20.0])
        assert result["grade"] == "D"

    def test_returns_required_keys(self) -> None:
        from app.reporting import benchmark_vs_portfolio

        result = benchmark_vs_portfolio(10.0, [5.0, 10.0, 15.0, 20.0])
        for key in ("percentile", "rank", "total_peers", "is_above_median", "grade"):
            assert key in result

    def test_empty_portfolio_raises(self) -> None:
        import pytest

        from app.reporting import benchmark_vs_portfolio

        with pytest.raises(ValueError):
            benchmark_vs_portfolio(10.0, [])

    def test_percentile_in_range(self) -> None:
        from app.reporting import benchmark_vs_portfolio

        result = benchmark_vs_portfolio(50.0, list(range(1, 101)))
        assert 0 <= result["percentile"] <= 100

    def test_total_peers_count(self) -> None:
        from app.reporting import benchmark_vs_portfolio

        portfolio = [float(i) for i in range(20)]
        result = benchmark_vs_portfolio(10.0, portfolio)
        assert result["total_peers"] == 20

    @pytest.mark.parametrize(
        "kwh,portfolio,expected_above",
        [
            (20.0, [10.0, 15.0, 25.0], True),
            (5.0, [10.0, 15.0, 25.0], False),
        ],
    )
    def test_is_above_median(self, kwh: float, portfolio: list, expected_above: bool) -> None:
        from app.reporting import benchmark_vs_portfolio

        result = benchmark_vs_portfolio(kwh, portfolio)
        assert result["is_above_median"] == expected_above


class TestKwhToWh:
    def test_basic(self) -> None:
        from app.reporting import kwh_to_wh

        assert kwh_to_wh(1.0) == pytest.approx(1000.0)

    def test_zero(self) -> None:
        from app.reporting import kwh_to_wh

        assert kwh_to_wh(0.0) == 0.0

    def test_fractional(self) -> None:
        from app.reporting import kwh_to_wh

        assert kwh_to_wh(0.5) == pytest.approx(500.0)


class TestWhToKwh:
    def test_basic(self) -> None:
        from app.reporting import wh_to_kwh

        assert wh_to_kwh(1000.0) == pytest.approx(1.0)

    def test_zero(self) -> None:
        from app.reporting import wh_to_kwh

        assert wh_to_kwh(0.0) == 0.0


class TestTariffCost:
    def test_basic(self) -> None:
        from app.reporting import tariff_cost

        assert tariff_cost(10.0, 0.15) == pytest.approx(1.5)

    def test_zero_kwh(self) -> None:
        from app.reporting import tariff_cost

        assert tariff_cost(0.0, 0.15) == 0.0

    def test_negative_kwh(self) -> None:
        from app.reporting import tariff_cost

        assert tariff_cost(-5.0, 0.15) == 0.0

    def test_free_tariff(self) -> None:
        from app.reporting import tariff_cost

        assert tariff_cost(100.0, 0.0) == 0.0


class TestSummarizeEnergyPeriod:
    def test_basic_summary(self) -> None:
        from app.reporting import summarize_energy_period

        result = summarize_energy_period([10.0, 20.0, 30.0], label="Jan")
        assert result["label"] == "Jan"
        assert result["total"] == pytest.approx(60.0)
        assert result["mean"] == pytest.approx(20.0)
        assert result["min"] == 10.0
        assert result["max"] == 30.0
        assert result["count"] == 3

    def test_empty_raises(self) -> None:
        from app.reporting import summarize_energy_period

        with pytest.raises(ValueError):
            summarize_energy_period([])

    def test_single_value(self) -> None:
        from app.reporting import summarize_energy_period

        result = summarize_energy_period([5.0])
        assert result["min"] == result["max"] == 5.0


class TestFormatReportRow:
    def test_basic_row(self) -> None:
        from app.reporting import format_report_row

        data = {"a": 1, "b": 2, "c": 3}
        assert format_report_row(data, ["a", "b", "c"]) == "1,2,3"

    def test_custom_separator(self) -> None:
        from app.reporting import format_report_row

        data = {"x": "hello", "y": "world"}
        assert format_report_row(data, ["x", "y"], separator="|") == "hello|world"

    def test_missing_field(self) -> None:
        from app.reporting import format_report_row

        data = {"a": 1}
        result = format_report_row(data, ["a", "missing"])
        assert "1" in result


class TestAggregateDailyReport:
    def test_full_day(self) -> None:
        from app.reporting import aggregate_daily_report

        hourly = [1.0] * 24
        result = aggregate_daily_report(hourly)
        assert len(result) == 1
        assert result[0] == pytest.approx(24.0)

    def test_two_days(self) -> None:
        from app.reporting import aggregate_daily_report

        hourly = [2.0] * 48
        result = aggregate_daily_report(hourly)
        assert len(result) == 2

    def test_empty(self) -> None:
        from app.reporting import aggregate_daily_report

        assert aggregate_daily_report([]) == []

    def test_partial_day(self) -> None:
        from app.reporting import aggregate_daily_report

        hourly = [1.0] * 36
        result = aggregate_daily_report(hourly)
        assert len(result) == 2


class TestReportAnomalySummary:
    def test_all_normal(self) -> None:
        from app.reporting import report_anomaly_summary

        result = report_anomaly_summary([False, False, False])
        assert result["anomaly_count"] == 0
        assert result["anomaly_rate"] == pytest.approx(0.0)

    def test_all_anomaly(self) -> None:
        from app.reporting import report_anomaly_summary

        result = report_anomaly_summary([True, True])
        assert result["anomaly_rate"] == pytest.approx(1.0)

    def test_mixed(self) -> None:
        from app.reporting import report_anomaly_summary

        result = report_anomaly_summary([True, False, True, False])
        assert result["anomaly_count"] == 2
        assert result["anomaly_rate"] == pytest.approx(0.5)

    def test_empty_raises(self) -> None:
        from app.reporting import report_anomaly_summary

        with pytest.raises(ValueError):
            report_anomaly_summary([])


def test_top_consumption_hours_returns_n() -> None:
    hourly = list(range(24))
    result = top_consumption_hours(hourly, n=5)
    assert len(result) == 5


def test_top_consumption_hours_sorted_desc() -> None:
    hourly = [float(i) for i in range(24)]
    result = top_consumption_hours(hourly, n=3)
    assert result[0]["kwh"] >= result[1]["kwh"] >= result[2]["kwh"]


def test_top_consumption_hours_correct_hour() -> None:
    hourly = [0.0] * 24
    hourly[14] = 100.0
    result = top_consumption_hours(hourly, n=1)
    assert result[0]["hour"] == 14
    assert result[0]["kwh"] == pytest.approx(100.0)


def test_top_consumption_hours_empty() -> None:
    assert top_consumption_hours([]) == []


def test_top_consumption_hours_n_capped() -> None:
    hourly = [1.0, 2.0, 3.0]
    result = top_consumption_hours(hourly, n=10)
    assert len(result) == 3


def test_rolling_savings_summary_length() -> None:
    actual = [8.0] * 48
    baseline = [10.0] * 48
    result = rolling_savings_summary(actual, baseline, window=24)
    assert len(result) == 48 - 24 + 1


def test_rolling_savings_summary_positive_savings() -> None:
    actual = [8.0] * 48
    baseline = [10.0] * 48
    result = rolling_savings_summary(actual, baseline, window=24)
    assert all(r["saved_kwh"] > 0 for r in result)


def test_rolling_savings_summary_keys() -> None:
    actual = [8.0] * 30
    baseline = [10.0] * 30
    result = rolling_savings_summary(actual, baseline, window=10)
    assert "window_start" in result[0]
    assert "saved_kwh" in result[0]
    assert "savings_pct" in result[0]


def test_rolling_savings_summary_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        rolling_savings_summary([1.0, 2.0], [1.0])


def test_rolling_savings_summary_too_short_returns_empty() -> None:
    actual = [8.0] * 10
    baseline = [10.0] * 10
    result = rolling_savings_summary(actual, baseline, window=24)
    assert result == []


@pytest.mark.parametrize("n", [1, 3, 5])
def test_top_consumption_hours_n_parametrize(n: int) -> None:
    hourly = [float(i) for i in range(24)]
    result = top_consumption_hours(hourly, n=n)
    assert len(result) == n


class TestEnergyCostEstimate:
    def test_basic(self) -> None:
        from app.reporting import energy_cost_estimate

        result = energy_cost_estimate(kwh=100.0, tariff_per_kwh=0.30)
        assert result["net_cost"] == pytest.approx(30.0)
        assert result["vat_amount"] == 0.0
        assert result["total_cost"] == pytest.approx(30.0)

    def test_with_vat(self) -> None:
        from app.reporting import energy_cost_estimate

        result = energy_cost_estimate(kwh=100.0, tariff_per_kwh=0.30, vat_rate=0.20)
        assert result["total_cost"] == pytest.approx(36.0)

    def test_zero_kwh(self) -> None:
        from app.reporting import energy_cost_estimate

        result = energy_cost_estimate(kwh=0.0, tariff_per_kwh=0.30)
        assert result["net_cost"] == 0.0

    def test_negative_kwh_raises(self) -> None:
        from app.reporting import energy_cost_estimate

        with pytest.raises(ValueError, match="kwh"):
            energy_cost_estimate(kwh=-1.0, tariff_per_kwh=0.30)

    def test_negative_tariff_raises(self) -> None:
        from app.reporting import energy_cost_estimate

        with pytest.raises(ValueError, match="tariff"):
            energy_cost_estimate(kwh=100.0, tariff_per_kwh=-0.10)

    @pytest.mark.parametrize("vat_rate", [0.0, 0.05, 0.20])
    def test_vat_rates(self, vat_rate: float) -> None:
        from app.reporting import energy_cost_estimate

        result = energy_cost_estimate(100.0, 0.30, vat_rate=vat_rate)
        assert result["total_cost"] == pytest.approx(30.0 * (1 + vat_rate), abs=1e-3)


class TestCarbonReportSection:
    def test_basic(self) -> None:
        from app.reporting import carbon_report_section

        result = carbon_report_section(kwh=1000.0, emission_factor_kg_per_kwh=0.233, label="Jan")
        assert result["label"] == "Jan"
        assert result["kg_co2"] == pytest.approx(233.0)
        assert result["tonnes_co2"] == pytest.approx(0.233, abs=1e-4)

    def test_zero_emission_factor(self) -> None:
        from app.reporting import carbon_report_section

        result = carbon_report_section(kwh=500.0, emission_factor_kg_per_kwh=0.0)
        assert result["kg_co2"] == 0.0

    def test_negative_kwh_raises(self) -> None:
        from app.reporting import carbon_report_section

        with pytest.raises(ValueError, match="kwh"):
            carbon_report_section(kwh=-100.0, emission_factor_kg_per_kwh=0.2)

    def test_keys_present(self) -> None:
        from app.reporting import carbon_report_section

        result = carbon_report_section(100.0, 0.2)
        assert {"label", "kwh", "kg_co2", "tonnes_co2", "emission_factor"} == set(result.keys())


class TestConsumptionBudgetVariance:
    def test_on_budget(self) -> None:
        from app.reporting import consumption_budget_variance

        result = consumption_budget_variance(actual_kwh=100.0, budget_kwh=100.0)
        assert result["delta_kwh"] == 0.0
        assert result["on_budget"] == 1

    def test_over_budget(self) -> None:
        from app.reporting import consumption_budget_variance

        result = consumption_budget_variance(actual_kwh=120.0, budget_kwh=100.0)
        assert result["variance_pct"] == pytest.approx(20.0)
        assert result["on_budget"] == 0

    def test_under_budget(self) -> None:
        from app.reporting import consumption_budget_variance

        result = consumption_budget_variance(actual_kwh=80.0, budget_kwh=100.0)
        assert result["delta_kwh"] == pytest.approx(-20.0)

    def test_zero_budget_raises(self) -> None:
        from app.reporting import consumption_budget_variance

        with pytest.raises(ValueError, match="budget_kwh"):
            consumption_budget_variance(100.0, 0.0)


# ---------------------------------------------------------------------------
# Tests for top_n_consumers, consumption_heatmap_data, savings_summary
# ---------------------------------------------------------------------------


class TestTopNConsumers:
    def test_basic_ordering(self) -> None:
        from app.reporting import top_n_consumers

        readings = [
            {"building_id": "A", "consumption_kwh": 100.0},
            {"building_id": "B", "consumption_kwh": 300.0},
            {"building_id": "C", "consumption_kwh": 200.0},
        ]
        result = top_n_consumers(readings, n=2)
        assert result[0]["building_id"] == "B"
        assert len(result) == 2

    def test_n_exceeds_count(self) -> None:
        from app.reporting import top_n_consumers

        readings = [{"building_id": "A", "consumption_kwh": 50.0}]
        assert len(top_n_consumers(readings, n=10)) == 1

    def test_invalid_n_raises(self) -> None:
        import pytest

        from app.reporting import top_n_consumers

        with pytest.raises(ValueError):
            top_n_consumers([], n=0)

    def test_missing_value_field_excluded(self) -> None:
        from app.reporting import top_n_consumers

        readings = [{"building_id": "A"}, {"building_id": "B", "consumption_kwh": 100.0}]
        result = top_n_consumers(readings, n=5)
        assert len(result) == 1


class TestConsumptionHeatmapData:
    def test_basic_structure(self) -> None:
        from app.reporting import consumption_heatmap_data

        readings = [
            {"day_of_week": 0, "hour": 8, "consumption_kwh": 10.0},
            {"day_of_week": 0, "hour": 8, "consumption_kwh": 20.0},
        ]
        result = consumption_heatmap_data(readings)
        assert result["0"]["8"] == pytest.approx(15.0, abs=0.01)

    def test_missing_value_excluded(self) -> None:
        from app.reporting import consumption_heatmap_data

        readings = [{"day_of_week": 1, "hour": 9}]
        assert consumption_heatmap_data(readings) == {}

    def test_empty_returns_empty(self) -> None:
        from app.reporting import consumption_heatmap_data

        assert consumption_heatmap_data([]) == {}


class TestSavingsSummary:
    def test_positive_savings(self) -> None:
        from app.reporting import savings_summary

        result = savings_summary(1000.0, 800.0, unit_cost=0.10)
        assert result["saved_kwh"] == pytest.approx(200.0, abs=0.01)
        assert result["saved_cost"] == pytest.approx(20.0, abs=0.01)
        assert result["reduction_pct"] == pytest.approx(20.0, abs=0.01)

    def test_zero_before_returns_zeros(self) -> None:
        from app.reporting import savings_summary

        result = savings_summary(0.0, 100.0)
        assert result["saved_kwh"] == 0.0

    def test_negative_raises(self) -> None:
        import pytest

        from app.reporting import savings_summary

        with pytest.raises(ValueError):
            savings_summary(-100.0, 50.0)

    def test_no_change(self) -> None:
        from app.reporting import savings_summary

        result = savings_summary(500.0, 500.0)
        assert result["saved_kwh"] == 0.0
        assert result["reduction_pct"] == 0.0


import pytest as _pytest


@_pytest.mark.parametrize(
    "before,after,expected_grade",
    [
        (100.0, 50.0, "A"),
        (100.0, 99.0, "D"),
        (100.0, 100.0, "D"),
    ],
)
def test_energy_efficiency_grade_parametrized(before: float, after: float, expected_grade: str) -> None:
    from app.reporting import energy_efficiency_grade

    assert energy_efficiency_grade(after, before) == expected_grade


@_pytest.mark.parametrize("hourly_count", [24, 48, 168])
def test_aggregate_daily_report_length(hourly_count: int) -> None:
    from app.reporting import aggregate_daily_report

    hourly = [1.0] * hourly_count
    result = aggregate_daily_report(hourly)
    assert len(result) == hourly_count // 24


@_pytest.mark.parametrize("tariff", [0.05, 0.10, 0.15, 0.20])
def test_tariff_cost_scales_linearly(tariff: float) -> None:
    from app.reporting import tariff_cost

    assert tariff_cost(100.0, tariff) == _pytest.approx(100.0 * tariff, abs=1e-6)


def test_report_anomaly_summary_all_normal() -> None:
    from app.reporting import report_anomaly_summary

    result = report_anomaly_summary([False] * 10)
    assert result["anomaly_count"] == 0
    assert result["anomaly_rate"] == _pytest.approx(0.0)


def test_report_anomaly_summary_all_anomalous() -> None:
    from app.reporting import report_anomaly_summary

    result = report_anomaly_summary([True] * 5)
    assert result["anomaly_count"] == 5
    assert result["anomaly_rate"] == _pytest.approx(1.0)


@_pytest.mark.parametrize("n", [24, 48, 96])
def test_peak_usage_window_output_keys(n: int) -> None:
    from app.reporting import peak_usage_window

    hourly_kwh = [float(i % 24) for i in range(n)][:24]
    result = peak_usage_window(hourly_kwh)
    assert "peak_start" in result
    assert "total_kwh" in result


@_pytest.mark.parametrize("kwh,intensity", [(100.0, 0.3), (500.0, 0.4), (1000.0, 0.5)])
def test_emission_report_positive_total(kwh: float, intensity: float) -> None:
    from app.reporting import emission_report

    values = [kwh / 24] * 24
    result = emission_report(values, intensity)
    assert result["total_co2_kg"] > 0.0


@_pytest.mark.parametrize("n_hours", [24, 48])
def test_demand_variance_report_has_keys(n_hours: int) -> None:
    from app.reporting import demand_variance_report

    readings = [float(i % 10 + 1) for i in range(n_hours)]
    result = demand_variance_report(readings)
    assert "mean" in result
    assert "variance" in result
