"""Ops-Vision: SRE ML platform for incident prediction and anomaly detection."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("ops-vision")
except PackageNotFoundError:
    __version__ = "1.0.0"

__all__ = ["__version__"]
