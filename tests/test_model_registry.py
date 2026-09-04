"""Tests for app.model_registry module."""

from __future__ import annotations

import pytest

from app.model_registry import ModelRegistry, ModelStage, ModelVersion


def _mv(name="price-model", version="1.0.0", path="s3://models/v1") -> ModelVersion:
    return ModelVersion(name=name, version=version, artifact_path=path)


class TestModelVersionRegistration:
    def test_register_succeeds(self):
        reg = ModelRegistry()
        reg.register(_mv())
        assert "price-model" in reg.list_models()

    def test_duplicate_version_raises(self):
        reg = ModelRegistry()
        reg.register(_mv())
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_mv())

    def test_different_versions_accepted(self):
        reg = ModelRegistry()
        reg.register(_mv(version="1.0.0"))
        reg.register(_mv(version="2.0.0"))
        assert len(reg.list_versions("price-model")) == 2

    def test_default_stage_is_staging(self):
        reg = ModelRegistry()
        reg.register(_mv())
        assert reg.get_latest("price-model").stage is ModelStage.STAGING


class TestStageTransitions:
    def test_transition_to_production(self):
        reg = ModelRegistry()
        reg.register(_mv())
        assert reg.transition_stage("price-model", "1.0.0", ModelStage.PRODUCTION)
        assert reg.get_production("price-model").version == "1.0.0"

    def test_transition_nonexistent_returns_false(self):
        reg = ModelRegistry()
        assert reg.transition_stage("ghost", "1.0.0", ModelStage.PRODUCTION) is False

    def test_get_production_none_when_none(self):
        reg = ModelRegistry()
        reg.register(_mv())
        assert reg.get_production("price-model") is None

    def test_archive_removes_from_production(self):
        reg = ModelRegistry()
        reg.register(_mv())
        reg.transition_stage("price-model", "1.0.0", ModelStage.PRODUCTION)
        reg.transition_stage("price-model", "1.0.0", ModelStage.ARCHIVED)
        assert reg.get_production("price-model") is None

    @pytest.mark.parametrize("stage", list(ModelStage))
    def test_all_stage_values_accepted(self, stage):
        reg = ModelRegistry()
        reg.register(_mv())
        assert reg.transition_stage("price-model", "1.0.0", stage) is True


class TestGetLatest:
    def test_latest_returns_last_registered(self):
        reg = ModelRegistry()
        reg.register(_mv(version="1.0.0"))
        reg.register(_mv(version="2.0.0"))
        assert reg.get_latest("price-model").version == "2.0.0"

    def test_get_latest_unknown_model_returns_none(self):
        assert ModelRegistry().get_latest("ghost") is None


class TestTagging:
    def test_add_tag_succeeds(self):
        reg = ModelRegistry()
        reg.register(_mv())
        assert reg.add_tag("price-model", "1.0.0", "team", "ml") is True
        mv = reg.get_latest("price-model")
        assert mv.tags["team"] == "ml"

    def test_add_tag_nonexistent_returns_false(self):
        reg = ModelRegistry()
        assert reg.add_tag("ghost", "1.0.0", "k", "v") is False

    def test_metrics_stored(self):
        mv = _mv()
        mv.metrics = {"rmse": 0.05, "mae": 0.03}
        reg = ModelRegistry()
        reg.register(mv)
        assert reg.get_latest("price-model").metrics["rmse"] == 0.05


class TestListModels:
    def test_empty_registry_list_empty(self):
        reg = ModelRegistry()
        assert reg.list_models() == []

    def test_multiple_model_names(self):
        reg = ModelRegistry()
        reg.register(_mv("model-a"))
        reg.register(_mv("model-b"))
        assert set(reg.list_models()) == {"model-a", "model-b"}

    def test_list_versions_empty_for_unknown_model(self):
        reg = ModelRegistry()
        assert reg.list_versions("no-such-model") == []


class TestProductionPromotion:
    def test_only_one_production_at_a_time(self):
        reg = ModelRegistry()
        reg.register(_mv(version="1.0.0"))
        reg.register(_mv(version="2.0.0"))
        reg.transition_stage("price-model", "1.0.0", ModelStage.PRODUCTION)
        reg.transition_stage("price-model", "2.0.0", ModelStage.PRODUCTION)
        prod = reg.get_production("price-model")
        assert prod.version == "2.0.0"

    def test_multiple_tags_per_version(self):
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
