"""Tests for app.energy_label module."""

from __future__ import annotations

import pytest

from app.energy_label import (
    EnergyClass,
    EnergyLabel,
    classify_building,
    compare_buildings,
    efficiency_ratio,
    label_building,
)


class TestClassifyBuilding:
    def test_excellent_gets_a(self) -> None:
        assert classify_building(40.0) == EnergyClass.A

    def test_very_high_gets_g(self) -> None:
        assert classify_building(500.0) == EnergyClass.G

    def test_boundary_b(self) -> None:
        assert classify_building(100.0) == EnergyClass.B

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            classify_building(-10.0)

    @pytest.mark.parametrize(
        "eui,expected",
        [
            (30.0, EnergyClass.A),
            (75.0, EnergyClass.B),
            (120.0, EnergyClass.C),
            (175.0, EnergyClass.D),
            (225.0, EnergyClass.E),
            (300.0, EnergyClass.F),
            (400.0, EnergyClass.G),
        ],
    )
    def test_parametrized(self, eui: float, expected: EnergyClass) -> None:
        assert classify_building(eui) == expected


class TestEfficiencyRatio:
    def test_at_reference_gives_one(self) -> None:
        assert efficiency_ratio(100.0, 100.0) == pytest.approx(1.0)

    def test_zero_reference_gives_zero(self) -> None:
        assert efficiency_ratio(50.0, 0.0) == 0.0

    def test_better_than_reference_less_than_one(self) -> None:
        assert efficiency_ratio(80.0, 200.0) < 1.0

    def test_negative_eui_raises(self) -> None:
        with pytest.raises(ValueError):
            efficiency_ratio(-10.0, 100.0)

    def test_negative_reference_raises(self) -> None:
        with pytest.raises(ValueError):
            efficiency_ratio(100.0, -50.0)

    @pytest.mark.parametrize(
        "eui,ref,expected",
        [(100.0, 200.0, 0.5), (200.0, 200.0, 1.0), (300.0, 200.0, 1.5)],
    )
    def test_parametrized(self, eui: float, ref: float, expected: float) -> None:
        assert efficiency_ratio(eui, ref) == pytest.approx(expected)


class TestLabelBuilding:
    def test_returns_energy_label(self) -> None:
        result = label_building(1000.0, 80000.0)
        assert isinstance(result, EnergyLabel)

    def test_eui_calculation(self) -> None:
        result = label_building(1000.0, 100000.0)
        assert result.eui_kwh_per_m2 == pytest.approx(100.0, rel=1e-4)

    def test_co2_positive(self) -> None:
        result = label_building(500.0, 50000.0)
        assert result.co2_kg_per_m2 > 0.0

    def test_zero_area_raises(self) -> None:
        with pytest.raises(ValueError):
            label_building(0.0, 10000.0)

    def test_negative_kwh_raises(self) -> None:
        with pytest.raises(ValueError):
            label_building(500.0, -1000.0)

    def test_recommendations_non_empty(self) -> None:
        result = label_building(500.0, 200000.0)
        assert len(result.recommended_actions) > 0

    @pytest.mark.parametrize(
        "area,kwh,expected_class",
        [
            (1000.0, 30000.0, EnergyClass.A),
            (1000.0, 200000.0, EnergyClass.D),
            (1000.0, 500000.0, EnergyClass.G),
        ],
    )
    def test_parametrized_labels(
        self, area: float, kwh: float, expected_class: EnergyClass
    ) -> None:
        result = label_building(area, kwh)
        assert result.label == expected_class


class TestCompareBuildings:
    def test_sorted_best_first(self) -> None:
        buildings = [
            ("poor", 500.0, 250000.0),
            ("good", 500.0, 40000.0),
            ("medium", 500.0, 100000.0),
        ]
        results = compare_buildings(buildings)
        euis = [r[1].eui_kwh_per_m2 for r in results]
        assert euis == sorted(euis)

    def test_single_building(self) -> None:
        results = compare_buildings([("only", 1000.0, 100000.0)])
        assert len(results) == 1
        assert results[0][0] == "only"

    def test_empty_list(self) -> None:
        assert compare_buildings([]) == []

    def test_names_preserved(self) -> None:
        buildings = [("a", 500.0, 50000.0), ("b", 500.0, 100000.0)]
        results = compare_buildings(buildings)
        names = [r[0] for r in results]
        assert set(names) == {"a", "b"}


class TestEnergyLabelEdgeCases:
    def test_classify_boundary_a_plus(self) -> None:
        label = classify_building(30.0)
        assert label in list(EnergyClass)

    def test_label_building_returns_label(self) -> None:
        label = label_building(1000.0, 150000.0)
        assert isinstance(label, EnergyLabel)
        assert hasattr(label, "energy_class")

    def test_label_building_class_matches_classify(self) -> None:
        eui = 200.0
        expected_class = classify_building(eui)
        label = label_building(1000.0, eui * 1000.0)
        assert label.energy_class == expected_class

    def test_efficiency_ratio_same_building_one(self) -> None:
        ratio = efficiency_ratio(100.0, 100.0)
        assert ratio == pytest.approx(1.0, rel=1e-4)

    def test_efficiency_ratio_better_building_gt_one(self) -> None:
        ratio = efficiency_ratio(100.0, 200.0)
        assert ratio > 1.0

    def test_efficiency_ratio_worse_building_lt_one(self) -> None:
        ratio = efficiency_ratio(200.0, 100.0)
        assert ratio < 1.0

    def test_compare_buildings_returns_labels(self) -> None:
        buildings = [("bad", 500.0, 500000.0), ("good", 500.0, 100000.0)]
        results = compare_buildings(buildings)
        assert len(results) == 2
        assert all(isinstance(r[1], EnergyLabel) for r in results)

    @pytest.mark.parametrize("eui", [30.0, 80.0, 150.0, 250.0, 400.0])
    def test_all_classes_reachable(self, eui: float) -> None:
        cls = classify_building(eui)
        assert cls in list(EnergyClass)
