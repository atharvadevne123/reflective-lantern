"""Version information for the Logistics-Flow API."""
from __future__ import annotations

__version__ = "1.0.0"
__api_version__ = "v1"
__description__ = "Last-mile delivery time prediction with drift monitoring."


def get_version_info() -> dict[str, str]:
    """Return a dict of version metadata for API responses."""
    return {
        "version": __version__,
        "api_version": __api_version__,
        "description": __description__,
    }
