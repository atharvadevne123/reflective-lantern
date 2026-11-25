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
    from app.version import (
        __api_version__,
        __description__,
        __version__,
        get_version_info,
    )

    info = get_version_info()
    assert info["version"] == __version__
    assert info["api_version"] == __api_version__
    assert info["description"] == __description__


def test_version_major_is_integer() -> None:
    """__version__ major component should be a valid integer."""
    from app.version import __version__

    major = __version__.split(".")[0]
    assert major.isdigit()


def test_api_version_contains_digit() -> None:
    """__api_version__ should contain a numeric version number."""
    import re

    from app.version import __api_version__

    assert re.search(r"\d+", __api_version__), f"No digit found in: {__api_version__}"


def test_get_version_info_returns_strings() -> None:
    """All values in get_version_info() should be strings."""
    from app.version import get_version_info

    info = get_version_info()
    for key, value in info.items():
        assert isinstance(value, str), f"Key '{key}' has non-string value: {value!r}"


def test_description_does_not_start_with_whitespace() -> None:
    """__description__ should not have leading whitespace."""
    from app.version import __description__

    assert __description__ == __description__.lstrip()


def test_version_patch_is_integer() -> None:
    """__version__ patch component should be a valid integer."""
    from app.version import __version__

    patch = __version__.split(".")[-1]
    assert patch.isdigit()
