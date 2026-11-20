"""Tests for app/energy_benchmark.py."""

from __future__ import annotations

import pytest

from app.energy_benchmark import (
    MIN_COHORT_SIZE,
    benchmark,
    energy_use_intensity,
    grade_from_score,
    percentile_rank,
    savings_potential,
    score_from_percentile,
)

COHORT = [80.0, 90.0, 110.0, 120.0, 150.0]


class TestEnergyUseIntensity:
    def test_divides_consumption_by_area(self) -> None:
        assert energy_use_intensity(100_000.0, 1_000.0) == pytest.approx(100.0)

    def test_larger_building_has_lower_intensity(self) -> None:
        assert energy_use_intensity(100_000.0, 2_000.0) < energy_use_intensity(100_000.0, 1_000.0)

    def test_zero_consumption_gives_zero_intensity(self) -> None:
        assert energy_use_intensity(0.0, 1_000.0) == 0.0

    def test_negative_consumption_rejected(self) -> None:
        with pytest.raises(ValueError, match="annual_kwh must be non-negative"):
            energy_use_intensity(-1.0, 1_000.0)

    @pytest.mark.parametrize("area", [0.0, -100.0])
    def test_non_positive_area_rejected(self, area: float) -> None:
        with pytest.raises(ValueError, match="floor_area_m2 must be positive"):
            energy_use_intensity(100_000.0, area)


class TestPercentileRank:
    def test_best_in_cohort_ranks_top(self) -> None:
        assert percentile_rank(10.0, COHORT) == pytest.approx(100.0)

    def test_worst_in_cohort_ranks_bottom(self) -> None:
        assert percentile_rank(999.0, COHORT) == pytest.approx(0.0)

    def test_lower_eui_ranks_higher(self) -> None:
        assert percentile_rank(85.0, COHORT) > percentile_rank(130.0, COHORT)

    def test_median_building_ranks_mid_cohort(self) -> None:
        assert percentile_rank(110.0, COHORT) == pytest.approx(50.0)

    def test_ties_split_the_difference(self) -> None:
        # Identical to every peer: beats none, ties all, lands at 50.
        assert percentile_rank(100.0, [100.0, 100.0, 100.0]) == pytest.approx(50.0)

    def test_rank_is_bounded(self) -> None:
        for eui in (0.0, 85.0, 110.0, 500.0):
            assert 0.0 <= percentile_rank(eui, COHORT) <= 100.0

    def test_empty_cohort_rejected(self) -> None:
        with pytest.raises(ValueError, match="cohort_euis must not be empty"):
            percentile_rank(100.0, [])


class TestScoreFromPercentile:
    def test_top_percentile_scores_100(self) -> None:
        assert score_from_percentile(100.0) == 100

    def test_bottom_percentile_clamps_to_one(self) -> None:
        assert score_from_percentile(0.0) == 1

    def test_midpoint_scores_50(self) -> None:
        assert score_from_percentile(50.0) == 50

    def test_score_is_always_in_range(self) -> None:
        for percentile in (0.0, 0.4, 33.3, 99.6, 100.0):
            assert 1 <= score_from_percentile(percentile) <= 100

    @pytest.mark.parametrize("percentile", [-0.1, 100.1, 200.0])
    def test_out_of_range_percentile_rejected(self, percentile: float) -> None:
        with pytest.raises(ValueError, match="percentile must be in 0-100"):
            score_from_percentile(percentile)


class TestGradeFromScore:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [(100, "A"), (90, "A"), (89, "B"), (75, "B"), (74, "C"), (50, "C"), (49, "D"), (25, "D"), (24, "F"), (1, "F")],
    )
    def test_grade_boundaries(self, score: int, expected: str) -> None:
        assert grade_from_score(score) == expected

    def test_grades_improve_monotonically(self) -> None:
        order = {"F": 0, "D": 1, "C": 2, "B": 3, "A": 4}
        grades = [order[grade_from_score(s)] for s in range(1, 101)]
        assert grades == sorted(grades)


class TestSavingsPotential:
    def test_reaching_a_lower_target_saves_energy(self) -> None:
        # Halving the EUI halves the consumption.
        assert savings_potential(100_000.0, 100.0, 50.0) == pytest.approx(50_000.0)

    def test_already_at_target_saves_nothing(self) -> None:
        assert savings_potential(100_000.0, 100.0, 100.0) == 0.0

    def test_already_beating_target_saves_nothing(self) -> None:
        assert savings_potential(100_000.0, 80.0, 100.0) == 0.0

    def test_more_ambitious_target_saves_more(self) -> None:
        assert savings_potential(100_000.0, 100.0, 40.0) > savings_potential(100_000.0, 100.0, 70.0)

    def test_zero_target_saves_everything(self) -> None:
        assert savings_potential(100_000.0, 100.0, 0.0) == pytest.approx(100_000.0)

    @pytest.mark.parametrize("eui", [0.0, -10.0])
    def test_non_positive_eui_rejected(self, eui: float) -> None:
        with pytest.raises(ValueError, match="eui must be positive"):
            savings_potential(100_000.0, eui, 50.0)

    def test_negative_target_rejected(self) -> None:
        with pytest.raises(ValueError, match="target_eui must be non-negative"):
            savings_potential(100_000.0, 100.0, -1.0)


class TestBenchmark:
    def test_efficient_building_grades_well(self) -> None:
        result = benchmark(50_000.0, 1_000.0, COHORT)
        assert result.grade == "A"
        assert result.percentile_rank == pytest.approx(100.0)

    def test_wasteful_building_grades_poorly(self) -> None:
        result = benchmark(200_000.0, 1_000.0, COHORT)
        assert result.grade == "F"
        assert result.score < 25

    def test_reports_cohort_context(self) -> None:
        result = benchmark(100_000.0, 1_000.0, COHORT)
        assert result.cohort_size == len(COHORT)
        assert result.cohort_median_eui == pytest.approx(110.0)

    def test_grade_matches_standalone_mapping(self) -> None:
        result = benchmark(100_000.0, 1_000.0, COHORT)
        assert result.grade == grade_from_score(result.score)

    def test_eui_matches_standalone_calculation(self) -> None:
        result = benchmark(120_000.0, 1_500.0, COHORT)
        assert result.eui == pytest.approx(energy_use_intensity(120_000.0, 1_500.0))

    def test_above_median_building_has_savings_potential(self) -> None:
        # 150 kWh/m2 sits above the cohort median of 110.
        result = benchmark(150_000.0, 1_000.0, COHORT)
        assert result.savings_potential_kwh > 0

    def test_below_median_building_has_no_savings_potential(self) -> None:
        result = benchmark(60_000.0, 1_000.0, COHORT)
        assert result.savings_potential_kwh == 0.0

    def test_undersized_cohort_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 3 peers"):
            benchmark(100_000.0, 1_000.0, [100.0, 110.0])

    def test_minimum_cohort_is_accepted(self) -> None:
        cohort = [90.0] * MIN_COHORT_SIZE
        assert benchmark(100_000.0, 1_000.0, cohort).cohort_size == MIN_COHORT_SIZE

    def test_invalid_area_propagates(self) -> None:
        with pytest.raises(ValueError, match="floor_area_m2 must be positive"):
            benchmark(100_000.0, 0.0, COHORT)
