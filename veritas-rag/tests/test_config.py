"""Tests for veritas-rag app/config.py."""

from __future__ import annotations

import pytest

from app.config import Settings, is_anthropic_backend, settings_summary


class TestSettings:
    def test_default_settings(self) -> None:
        s = Settings()
        assert s is not None

    def test_default_chunk_size(self) -> None:
        s = Settings()
        assert s.chunk_size > 0

    def test_default_embedding_dim(self) -> None:
        s = Settings()
        assert s.embedding_dim > 0

    def test_default_generator_backend(self) -> None:
        s = Settings()
        assert s.generator_backend in ("extractive", "anthropic")

    def test_chunk_overlap_less_than_chunk_size(self) -> None:
        s = Settings()
        assert s.chunk_overlap < s.chunk_size

    def test_confidence_threshold_range(self) -> None:
        s = Settings()
        assert 0.0 < s.confidence_threshold <= 1.0

    def test_invalid_embedding_dim(self) -> None:
        with pytest.raises(ValueError):
            Settings(embedding_dim=0)

    def test_invalid_confidence_threshold(self) -> None:
        with pytest.raises(ValueError):
            Settings(confidence_threshold=0.0)

    def test_invalid_chunk_overlap(self) -> None:
        with pytest.raises(ValueError):
            Settings(chunk_size=100, chunk_overlap=100)

    def test_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("CHUNK_SIZE", "512")
        s = Settings()
        assert s.chunk_size == 512


class TestSettingsSummary:
    def test_returns_dict(self) -> None:
        s = Settings()
        result = settings_summary(s)
        assert isinstance(result, dict)

    def test_has_embedding_dim(self) -> None:
        s = Settings()
        assert "embedding_dim" in settings_summary(s)

    def test_has_generator_backend(self) -> None:
        s = Settings()
        assert "generator_backend" in settings_summary(s)

    def test_has_all_expected_keys(self) -> None:
        s = Settings()
        summary = settings_summary(s)
        expected = {
            "embedding_dim", "chunk_size", "chunk_overlap",
            "bm25_top_k", "vector_top_k", "final_top_k",
            "confidence_threshold", "chunk_trust_floor",
            "generator_backend", "near_dup_threshold",
        }
        assert expected.issubset(summary.keys())


class TestSettingsValidation:
    def test_invalid_near_dup_threshold(self) -> None:
        with pytest.raises(ValueError):
            Settings(near_dup_threshold=0.0)

    def test_invalid_freshness_half_life(self) -> None:
        with pytest.raises(ValueError):
            Settings(freshness_half_life_days=0.0)

    def test_invalid_final_top_k(self) -> None:
        with pytest.raises(ValueError):
            Settings(final_top_k=0)

    def test_negative_chunk_overlap_allowed(self) -> None:
        with pytest.raises(ValueError):
            Settings(chunk_overlap=-1)

    @pytest.mark.parametrize("backend", ["extractive", "anthropic"])
    def test_valid_generator_backends(self, backend: str) -> None:
        s = Settings(generator_backend=backend)
        assert s.generator_backend == backend


class TestIsAnthropicBackend:
    def test_anthropic_returns_true(self) -> None:
        s = Settings(generator_backend="anthropic")
        assert is_anthropic_backend(s) is True

    def test_extractive_returns_false(self) -> None:
        s = Settings(generator_backend="extractive")
        assert is_anthropic_backend(s) is False

    def test_summary_has_all_expected_keys_plus_generator(self) -> None:
        s = Settings()
        summary = settings_summary(s)
        assert "generator_backend" in summary

    def test_cache_ttl_default_positive(self) -> None:
        s = Settings()
        assert s.cache_ttl_seconds > 0.0

    def test_trace_buffer_default_positive(self) -> None:
        s = Settings()
        assert s.trace_buffer_size > 0
