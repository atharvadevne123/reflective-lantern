"""Version constants for Property-Sage."""

from __future__ import annotations

__version__ = "1.2.0"
__api_version__ = "v1"
__min_python__ = (3, 10)


def get_version() -> str:
    """Return the current package version string.

    Returns:
        Semantic version string.
    """
    return __version__


def get_api_version() -> str:
    """Return the current API version prefix.

    Returns:
        API version string (e.g. 'v1').
    """
    return __api_version__
