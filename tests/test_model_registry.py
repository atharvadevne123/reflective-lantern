"""Tests for app.model_registry module."""

from __future__ import annotations

import pytest

from app.model_registry import ModelRegistry, ModelStage, ModelVersion


def _mv(name="price-model", version="1.0.0", path="s3://models/v1") -> ModelVersion:
    return ModelVersion(name=name, version=version, artifact_path=path)


class TestModelVersionRegistration:
    def test_register_succeeds(self) -> None:
        reg = ModelRegistry()
        reg.register(_mv())
        assert "price-model" in reg.list_models()

    def test_duplicate_version_raises(self) -> None:
        reg = ModelRegistry()
        reg.register(_mv())
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_mv())

    def test_different_versions_accepted(self) -> None:
        reg = ModelRegistry()
        reg.register(_mv(version="1.0.0"))
        reg.register(_mv(version="2.0.0"))
        assert len(reg.list_versions("price-model")) == 2

    def test_default_stage_is_staging(self) -> None:
        reg = ModelRegistry()
        reg.register(_mv())
        assert reg.get_latest("price-model").stage is ModelStage.STAGING


class TestStageTransitions:
    def test_transition_to_production(self) -> None:
        reg = ModelRegistry()
        reg.register(_mv())
        assert reg.transition_stage("price-model", "1.0.0", ModelStage.PRODUCTION)
        assert reg.get_production("price-model").version == "1.0.0"

    def test_transition_nonexistent_returns_false(self) -> None:
        reg = ModelRegistry()
        assert reg.transition_stage("ghost", "1.0.0", ModelStage.PRODUCTION) is False

    def test_get_production_none_when_none(self) -> None:
        reg = ModelRegistry()
        reg.register(_mv())
        assert reg.get_production("price-model") is None

    def test_archive_removes_from_production(self) -> None:
        reg = ModelRegistry()
        reg.register(_mv())
        reg.transition_stage("price-model", "1.0.0", ModelStage.PRODUCTION)
        reg.transition_stage("price-model", "1.0.0", ModelStage.ARCHIVED)
        assert reg.get_production("price-model") is None

    @pytest.mark.parametrize("stage", list(ModelStage))
    def test_all_stage_values_accepted(self, stage) -> None:
        reg = ModelRegistry()
        reg.register(_mv())
        assert reg.transition_stage("price-model", "1.0.0", stage) is True


class TestGetLatest:
    def test_latest_returns_last_registered(self) -> None:
        reg = ModelRegistry()
        reg.register(_mv(version="1.0.0"))
        reg.register(_mv(version="2.0.0"))
        assert reg.get_latest("price-model").version == "2.0.0"

    def test_get_latest_unknown_model_returns_none(self) -> None:
        assert ModelRegistry().get_latest("ghost") is None


class TestTagging:
    def test_add_tag_succeeds(self) -> None:
        reg = ModelRegistry()
        reg.register(_mv())
        assert reg.add_tag("price-model", "1.0.0", "team", "ml") is True
        mv = reg.get_latest("price-model")
        assert mv.tags["team"] == "ml"

    def test_add_tag_nonexistent_returns_false(self) -> None:
        reg = ModelRegistry()
        assert reg.add_tag("ghost", "1.0.0", "k", "v") is False

    def test_metrics_stored(self) -> None:
        mv = _mv()
        mv.metrics = {"rmse": 0.05, "mae": 0.03}
        reg = ModelRegistry()
        reg.register(mv)
        assert reg.get_latest("price-model").metrics["rmse"] == 0.05


class TestListModels:
    def test_empty_registry_list_empty(self) -> None:
        reg = ModelRegistry()
        assert reg.list_models() == []

    def test_multiple_model_names(self) -> None:
        reg = ModelRegistry()
        reg.register(_mv("model-a"))
        reg.register(_mv("model-b"))
        assert set(reg.list_models()) == {"model-a", "model-b"}

    def test_list_versions_empty_for_unknown_model(self) -> None:
        reg = ModelRegistry()
        assert reg.list_versions("no-such-model") == []


class TestProductionPromotion:
    def test_only_one_production_at_a_time(self) -> None:
        reg = ModelRegistry()
        reg.register(_mv(version="1.0.0"))
        reg.register(_mv(version="2.0.0"))
        reg.transition_stage("price-model", "1.0.0", ModelStage.PRODUCTION)
        reg.transition_stage("price-model", "2.0.0", ModelStage.PRODUCTION)
        prod = reg.get_production("price-model")
        assert prod.version == "2.0.0"

    def test_multiple_tags_per_version(self) -> None:
        reg = ModelRegistry()
        reg.register(_mv())
        reg.add_tag("price-model", "1.0.0", "env", "prod")
        reg.add_tag("price-model", "1.0.0", "team", "ml")
        mv = reg.get_latest("price-model")
        assert mv.tags["env"] == "prod"
        assert mv.tags["team"] == "ml"

    @pytest.mark.parametrize("version", ["1.0.0", "2.0.0-beta", "0.0.1", "10.0.0"])
    def test_semver_like_versions_accepted(self, version: str) -> None:
        reg = ModelRegistry()
        reg.register(_mv(version=version))
        assert reg.get_latest("price-model").version == version


@pytest.mark.parametrize("n_versions", [1, 3, 5])
def test_list_versions_returns_all_registered(n_versions: int) -> None:
    """list_versions includes every version that was registered."""
    from app.model_registry import ModelRegistry, ModelVersion

    reg = ModelRegistry()
    for i in range(n_versions):
        reg.register(ModelVersion(name="model", version=f"{i}.0.0", metrics={}))
    assert len(reg.list_versions("model")) == n_versions


@pytest.mark.parametrize("n_models", [1, 3, 5])
def test_list_models_returns_all_registered_names(n_models: int) -> None:
    """list_models includes every distinct model name registered."""
    from app.model_registry import ModelRegistry, ModelVersion

    reg = ModelRegistry()
    for i in range(n_models):
        reg.register(ModelVersion(name=f"model_{i}", version="1.0.0", metrics={}))
    names = reg.list_models()
    assert all(f"model_{i}" in names for i in range(n_models))


class TestModelVersionEdgeCases:
    def test_metrics_dict_preserved(self) -> None:
        from app.model_registry import ModelRegistry, ModelVersion

        metrics = {"rmse": 0.5, "r2": 0.9}
        reg = ModelRegistry()
        reg.register(ModelVersion(name="mv_test", version="1.0.0", metrics=metrics))
        stored = reg.get_latest("mv_test")
        assert stored.metrics == metrics

    @pytest.mark.parametrize("stage_name", ["staging", "production", "archived"])
    def test_stage_transition_accepted(self, stage_name: str) -> None:
        from app.model_registry import ModelRegistry, ModelStage, ModelVersion

        reg = ModelRegistry()
        reg.register(ModelVersion(name="staged_model", version="1.0.0", metrics={}))
        target = ModelStage[stage_name.upper()]
        reg.transition_stage("staged_model", "1.0.0", target)
        stored = reg.get_latest("staged_model")
        assert stored.stage == target


class TestDeleteVersion:
    def test_delete_existing_version(self) -> None:
        from app.model_registry import ModelRegistry, ModelVersion

        reg = ModelRegistry()
        reg.register(ModelVersion(name="m", version="1.0.0", metrics={}))
        assert reg.delete_version("m", "1.0.0") is True
        assert reg.get_latest("m") is None

    def test_delete_nonexistent_returns_false(self) -> None:
        from app.model_registry import ModelRegistry

        reg = ModelRegistry()
        assert reg.delete_version("ghost", "1.0.0") is False

    def test_delete_leaves_other_versions(self) -> None:
        from app.model_registry import ModelRegistry, ModelVersion

        reg = ModelRegistry()
        reg.register(ModelVersion(name="m", version="1.0.0", metrics={}))
        reg.register(ModelVersion(name="m", version="2.0.0", metrics={}))
        reg.delete_version("m", "1.0.0")
        assert len(reg.list_versions("m")) == 1
        assert reg.list_versions("m")[0].version == "2.0.0"


class TestGetByStage:
    def test_empty_result_when_no_production(self) -> None:
        from app.model_registry import ModelRegistry, ModelStage, ModelVersion

        reg = ModelRegistry()
        reg.register(ModelVersion(name="m", version="1.0.0", metrics={}))
        result = reg.get_by_stage("m", ModelStage.PRODUCTION)
        assert result == []

    def test_returns_production_versions(self) -> None:
        from app.model_registry import ModelRegistry, ModelStage, ModelVersion

        reg = ModelRegistry()
        reg.register(ModelVersion(name="m", version="1.0.0", metrics={}))
        reg.transition_stage("m", "1.0.0", ModelStage.PRODUCTION)
        result = reg.get_by_stage("m", ModelStage.PRODUCTION)
        assert len(result) == 1
        assert result[0].version == "1.0.0"

    def test_returns_empty_for_unknown_model(self) -> None:
        from app.model_registry import ModelRegistry, ModelStage

        reg = ModelRegistry()
        assert reg.get_by_stage("unknown", ModelStage.STAGING) == []


class TestVersionCount:
    def test_zero_for_unknown_model(self) -> None:
        from app.model_registry import ModelRegistry

        reg = ModelRegistry()
        assert reg.version_count("unknown") == 0

    def test_counts_registered_versions(self) -> None:
        from app.model_registry import ModelRegistry, ModelVersion

        reg = ModelRegistry()
        reg.register(ModelVersion(name="m", version="1.0.0", metrics={}))
        reg.register(ModelVersion(name="m", version="2.0.0", metrics={}))
        assert reg.version_count("m") == 2

    @pytest.mark.parametrize("n", [1, 3, 5])
    def test_count_matches_registers(self, n: int) -> None:
        from app.model_registry import ModelRegistry, ModelVersion

        reg = ModelRegistry()
        for i in range(n):
            reg.register(ModelVersion(name="m", version=f"{i}.0.0", metrics={}))
        assert reg.version_count("m") == n


class TestRegisteredModelNames:
    def test_empty_registry_returns_empty(self) -> None:
        from app.model_registry import ModelRegistry

        reg = ModelRegistry()
        assert reg.registered_model_names() == []

    def test_single_model_listed(self) -> None:
        from app.model_registry import ModelRegistry, ModelVersion

        reg = ModelRegistry()
        reg.register(ModelVersion(name="alpha", version="1.0.0", metrics={}))
        assert reg.registered_model_names() == ["alpha"]

    def test_names_are_sorted(self) -> None:
        from app.model_registry import ModelRegistry, ModelVersion

        reg = ModelRegistry()
        for name in ["charlie", "alpha", "beta"]:
            reg.register(ModelVersion(name=name, version="1.0.0", metrics={}))
        assert reg.registered_model_names() == ["alpha", "beta", "charlie"]

    def test_duplicate_versions_count_once(self) -> None:
        from app.model_registry import ModelRegistry, ModelVersion

        reg = ModelRegistry()
        reg.register(ModelVersion(name="m", version="1.0.0", metrics={}))
        reg.register(ModelVersion(name="m", version="2.0.0", metrics={}))
        assert reg.registered_model_names() == ["m"]

    @pytest.mark.parametrize("n", [1, 3, 5])
    def test_name_count_matches_unique_models(self, n: int) -> None:
        from app.model_registry import ModelRegistry, ModelVersion

        reg = ModelRegistry()
        for i in range(n):
            reg.register(ModelVersion(name=f"model_{i}", version="1.0.0", metrics={}))
        assert len(reg.registered_model_names()) == n
