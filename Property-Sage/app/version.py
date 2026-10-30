"""Package version for Property-Sage."""
from __future__ import annotations

VERSION: str = "1.2.0"
VERSION_INFO: tuple[int, int, int] = (1, 2, 0)


def get_version() -> str:
    """Return the current package version string.

    Returns:
        Version string in 'MAJOR.MINOR.PATCH' format.
    """
    return VERSION
