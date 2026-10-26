"""Tests for app/version.py version metadata module."""

from __future__ import annotations


def test_version_string_is_semver() -> None:
    """__version__ should follow SemVer X.Y.Z format."""
    import re

    from app.version import __version__

    assert re.match(r"^\d+\.\d+\.\d+$", __version__), f"Not SemVer: {__version__}"


def test_api_version_starts_with_v() -> None:
    """__api_version__ should begin with 'v'."""
    from app.version import __api_version__

    assert __api_version__.startswith("v"), f"Expected 'v' prefix, got: {__api_version__}"


def test_description_is_nonempty_string() -> None:
    """__description__ should be a non-empty string."""
    from app.version import __description__

    assert isinstance(__description__, str)
    assert len(__description__) > 0


def test_get_version_info_returns_dict() -> None:
    """get_version_info() should return a dict with the expected keys."""
    from app.version import get_version_info

    info = get_version_info()
    assert isinstance(info, dict)
    assert "version" in info
    assert "api_version" in info
    assert "description" in info


def test_get_version_info_values_match_module_attrs() -> None:
    """Values returned by get_version_info() should match module-level attributes."""
    from app.version import __api_version__, __description__, __version__, get_version_info

    info = get_version_info()
    assert info["version"] == __version__
    assert info["api_version"] == __api_version__
    assert info["description"] == __description__
