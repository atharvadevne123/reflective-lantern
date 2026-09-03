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


class TestExperimentTrackerExtensions:
    def test_reset_counts_zeroes_all(self):
        exp = Experiment("e", [CONTROL, TREATMENT])
        for i in range(10):
            exp.assign(str(i))
        exp.reset_counts()
        assert exp.total_assignments() == 0

    def test_total_assignments_accumulates(self):
        exp = Experiment("e", [CONTROL])
        for i in range(5):
            exp.assign(str(i))
        assert exp.total_assignments() == 5

    def test_registry_deregister(self):
        reg = ExperimentRegistry()
        reg.register(Experiment("x", [CONTROL]))
        assert reg.deregister("x") is True
        assert reg.get("x") is None

    def test_registry_deregister_missing(self):
        reg = ExperimentRegistry()
        assert reg.deregister("ghost") is False

    def test_registry_active_experiments(self):
        reg = ExperimentRegistry()
        reg.register(Experiment("on", [CONTROL], enabled=True))
        reg.register(Experiment("off", [CONTROL], enabled=False))
        assert reg.active_experiments() == ["on"]

    def test_registry_len(self):
        reg = ExperimentRegistry()
        assert len(reg) == 0
        reg.register(Experiment("a", [CONTROL]))
        assert len(reg) == 1
