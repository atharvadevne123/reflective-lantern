"""Tests for Fleet-Mind version module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.version import __api_version__, __model_version__, __version__, version_info


def test_version_strings_are_non_empty():
    assert __version__
    assert __model_version__
    assert __api_version__


def test_version_info_returns_dict():
    info = version_info()
    assert isinstance(info, dict)


def test_version_info_has_required_keys():
    info = version_info()
    assert "package" in info
    assert "model" in info
    assert "api" in info


def test_version_info_values_match_module_constants():
    info = version_info()
    assert info["package"] == __version__
    assert info["model"] == __model_version__
    assert info["api"] == __api_version__


@pytest.mark.parametrize("key", ["package", "model", "api"])
def test_version_info_values_are_strings(key):
    info = version_info()
    assert isinstance(info[key], str)


def test_version_string_format():
    parts = __version__.split(".")
    assert len(parts) == 3
    for part in parts:
        assert part.isdigit()
