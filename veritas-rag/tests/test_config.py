"""Tests for veritas-rag app/config.py."""

from __future__ import annotations

import pytest

from app.config import Settings, settings_summary


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
