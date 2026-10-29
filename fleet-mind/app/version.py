"""Version information for the Fleet-Mind package."""

from __future__ import annotations

__version__ = "1.0.0"
__model_version__ = "1.0.0"
__api_version__ = "1.0.0"


def version_info() -> dict[str, str]:
    """Return a dict of all version strings.

    Returns:
        Dict with 'package', 'model', and 'api' version strings.
    """
    return {
        "package": __version__,
        "model": __model_version__,
        "api": __api_version__,
    }
