"""Tests for Settings validation in app/config.py."""

from __future__ import annotations

import pytest

from app.config import Settings


class TestSettingsDefaults:
    def test_default_settings_instantiate(self) -> None:
        """Settings() with no arguments uses sensible defaults."""
        s = Settings()
        assert s.embedding_dim > 0
        assert s.chunk_size > 0
        assert s.chunk_overlap < s.chunk_size
        assert 0.0 < s.confidence_threshold <= 1.0

    def test_api_prefix_default(self) -> None:
        s = Settings()
        assert s.api_prefix == "/api/v1"

    def test_generator_backend_default(self) -> None:
        s = Settings()
        assert s.generator_backend == "extractive"


class TestSettingsValidation:
    def test_zero_embedding_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="embedding_dim"):
            Settings(embedding_dim=0)

    def test_negative_embedding_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="embedding_dim"):
            Settings(embedding_dim=-1)

    def test_zero_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            Settings(chunk_size=0)

    def test_negative_chunk_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            Settings(chunk_overlap=-1)

    def test_overlap_ge_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            Settings(chunk_size=100, chunk_overlap=100)

    def test_confidence_threshold_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence_threshold"):
            Settings(confidence_threshold=0.0)

    def test_confidence_threshold_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence_threshold"):
            Settings(confidence_threshold=1.1)

    def test_chunk_trust_floor_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_trust_floor"):
            Settings(chunk_trust_floor=-0.1)

    def test_chunk_trust_floor_one_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_trust_floor"):
            Settings(chunk_trust_floor=1.0)

    def test_negative_freshness_half_life_raises(self) -> None:
        with pytest.raises(ValueError, match="freshness_half_life_days"):
            Settings(freshness_half_life_days=0.0)

    def test_near_dup_threshold_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="near_dup_threshold"):
            Settings(near_dup_threshold=0.0)

    def test_final_top_k_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="final_top_k"):
            Settings(final_top_k=0)


class TestSettingsCustomValues:
    def test_custom_valid_settings(self) -> None:
        """Custom values that pass validation are stored correctly."""
        s = Settings(
            embedding_dim=128,
            chunk_size=400,
            chunk_overlap=50,
            confidence_threshold=0.6,
            final_top_k=3,
        )
        assert s.embedding_dim == 128
        assert s.chunk_size == 400
        assert s.chunk_overlap == 50
        assert s.confidence_threshold == 0.6
        assert s.final_top_k == 3

    def test_boundary_confidence_threshold_one(self) -> None:
        """confidence_threshold=1.0 is a valid boundary value."""
        s = Settings(confidence_threshold=1.0)
        assert s.confidence_threshold == 1.0

    def test_boundary_chunk_trust_floor_zero(self) -> None:
        """chunk_trust_floor=0.0 is a valid boundary value."""
        s = Settings(chunk_trust_floor=0.0)
        assert s.chunk_trust_floor == 0.0


class TestSettingsSummary:
    def test_returns_expected_keys(self) -> None:
        from app.config import Settings, settings_summary

        s = Settings()
        summary = settings_summary(s)
        for key in (
            "embedding_dim",
            "chunk_size",
            "final_top_k",
            "confidence_threshold",
            "generator_backend",
        ):
            assert key in summary

    def test_values_match_settings(self) -> None:
        from app.config import Settings, settings_summary

        s = Settings(embedding_dim=128, final_top_k=3)
        summary = settings_summary(s)
        assert summary["embedding_dim"] == 128
        assert summary["final_top_k"] == 3

    def test_summary_has_near_dup_threshold(self) -> None:
        from app.config import Settings, settings_summary

        s = Settings(near_dup_threshold=0.90)
        summary = settings_summary(s)
        assert summary["near_dup_threshold"] == pytest.approx(0.90)
