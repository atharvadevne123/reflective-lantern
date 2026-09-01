"""Tests for app.experiment_tracker module."""

from __future__ import annotations

import pytest

from app.experiment_tracker import (
    Experiment,
    ExperimentRegistry,
    Variant,
)

CONTROL = Variant("control", weight=1.0)
TREATMENT = Variant("treatment", weight=1.0)


class TestExperiment:
    def test_assignment_is_deterministic(self):
        exp = Experiment("test", [CONTROL, TREATMENT])
        r1 = exp.assign("user-42")
        r2 = exp.assign("user-42")
        assert r1.variant == r2.variant

    def test_different_entities_can_get_different_variants(self):
        exp = Experiment("test", [CONTROL, TREATMENT])
        variants = {exp.assign(str(i)).variant for i in range(100)}
        assert len(variants) == 2

    def test_disabled_experiment_returns_first_variant(self):
        exp = Experiment("test", [CONTROL, TREATMENT], enabled=False)
        result = exp.assign("any-user")
        assert result.variant == "control"

    def test_empty_variants_raises(self):
        with pytest.raises(ValueError, match="variant"):
            Experiment("test", [])

    def test_result_fields(self):
        exp = Experiment("my-exp", [CONTROL])
        result = exp.assign("u1")
        assert result.experiment == "my-exp"
        assert result.entity_id == "u1"

    def test_assignment_distribution_counted(self):
        exp = Experiment("dist", [CONTROL, TREATMENT])
        for i in range(1000):
            exp.assign(str(i))
        dist = exp.assignment_distribution()
        assert "control" in dist and "treatment" in dist
        assert sum(dist.values()) == 1000

    def test_weighted_variant_biases_assignment(self):
        heavy = Variant("heavy", weight=9.0)
        light = Variant("light", weight=1.0)
        exp = Experiment("weighted", [heavy, light])
        results = [exp.assign(str(i)).variant for i in range(500)]
        heavy_count = results.count("heavy")
        assert heavy_count > 300  # roughly 90%

    @pytest.mark.parametrize("entity_id", ["a", "user-999", "abc123"])
    def test_various_entity_ids_assigned(self, entity_id):
        exp = Experiment("t", [CONTROL, TREATMENT])
        r = exp.assign(entity_id)
        assert r.variant in ("control", "treatment")


class TestExperimentRegistry:
    def test_register_and_retrieve(self):
        reg = ExperimentRegistry()
        exp = Experiment("x", [CONTROL])
        reg.register(exp)
        assert reg.get("x") is exp

    def test_assign_via_registry(self):
        reg = ExperimentRegistry()
        reg.register(Experiment("r", [CONTROL, TREATMENT]))
        result = reg.assign("r", "user-1")
        assert result is not None
        assert result.variant in ("control", "treatment")

    def test_assign_unknown_experiment_returns_none(self):
        reg = ExperimentRegistry()
        assert reg.assign("ghost", "user") is None

    def test_list_experiments(self):
        reg = ExperimentRegistry()
        reg.register(Experiment("a", [CONTROL]))
        reg.register(Experiment("b", [CONTROL]))
        assert set(reg.list_experiments()) == {"a", "b"}

    def test_register_duplicate_overwrites(self) -> None:
        reg = ExperimentRegistry()
        exp1 = Experiment("dup", [CONTROL])
        exp2 = Experiment("dup", [CONTROL, TREATMENT])
        reg.register(exp1)
        reg.register(exp2)
        assert reg.get("dup") is exp2

    def test_empty_registry_list_is_empty(self) -> None:
        reg = ExperimentRegistry()
        assert reg.list_experiments() == []


@pytest.mark.parametrize("weight", [0.1, 1.0, 10.0])
def test_variant_weight_stored(weight: float) -> None:
    v = Variant("v", weight=weight)
    assert v.weight == weight
