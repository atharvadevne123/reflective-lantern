"""Tests for veritas-rag AnthropicGenerator with mocked Anthropic client."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas import Chunk, ScoredChunk


def _make_scored(text: str, trust: float = 0.8) -> ScoredChunk:
    chunk = Chunk(
        chunk_id="d1:1:0",
        doc_id="d1",
        source="test.pdf",
        source_type="pdf",
        page=1,
        seq=0,
        text=text,
        created_at="2026-01-01T00:00:00",
    )
    return ScoredChunk(chunk=chunk, trust=trust)


def _mock_anthropic():
    mock_anthropic_mod = MagicMock()
    mock_client = MagicMock()
    mock_anthropic_mod.Anthropic.return_value = mock_client
    return mock_anthropic_mod, mock_client


def test_anthropic_generator_init_with_mock():
    mock_mod, mock_client = _mock_anthropic()
    with patch.dict("sys.modules", {"anthropic": mock_mod}):
        from app.generation.llm import AnthropicGenerator
        gen = AnthropicGenerator()
        assert gen.model is not None
        assert gen.max_tokens > 0


def test_anthropic_generator_default_model():
    mock_mod, _ = _mock_anthropic()
    with patch.dict("sys.modules", {"anthropic": mock_mod}):
        from app.generation.llm import AnthropicGenerator
        gen = AnthropicGenerator()
        assert isinstance(gen.model, str)
        assert len(gen.model) > 0


def test_anthropic_generator_custom_model():
    mock_mod, _ = _mock_anthropic()
    with patch.dict("sys.modules", {"anthropic": mock_mod}):
        from app.generation.llm import AnthropicGenerator
        gen = AnthropicGenerator(model="claude-haiku-5-5", max_tokens=512)
        assert gen.model == "claude-haiku-5-5"
        assert gen.max_tokens == 512


def test_generate_returns_tuple_on_success():
    mock_mod, mock_client = _mock_anthropic()
    mock_response = MagicMock()
    mock_content_block = MagicMock()
    mock_content_block.type = "text"
    mock_content_block.text = "Energy consumption is high."
    mock_response.content = [mock_content_block]
    mock_client.messages.create.return_value = mock_response

    with patch.dict("sys.modules", {"anthropic": mock_mod}):
        from app.generation.llm import AnthropicGenerator
        gen = AnthropicGenerator()
        answer, citations = gen.generate("What is energy?", [_make_scored("Energy is power.")])
        assert isinstance(answer, str)
        assert isinstance(citations, list)


def test_generate_empty_chunks_returns_empty():
    mock_mod, mock_client = _mock_anthropic()
    mock_response = MagicMock()
    mock_response.content = []
    mock_client.messages.create.return_value = mock_response

    with patch.dict("sys.modules", {"anthropic": mock_mod}):
        from app.generation.llm import AnthropicGenerator
        gen = AnthropicGenerator()
        answer, citations = gen.generate("test query", [])
        assert isinstance(answer, str)
        assert isinstance(citations, list)


def test_generate_handles_api_exception():
    mock_mod, mock_client = _mock_anthropic()
    mock_client.messages.create.side_effect = Exception("API error")

    with patch.dict("sys.modules", {"anthropic": mock_mod}):
        from app.generation.llm import AnthropicGenerator
        gen = AnthropicGenerator()
        answer, citations = gen.generate("test", [_make_scored("text")])
        assert isinstance(answer, str)
        assert isinstance(citations, list)
