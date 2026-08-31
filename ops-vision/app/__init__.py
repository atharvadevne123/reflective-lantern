"""Ops-Vision: SRE ML platform for incident prediction and anomaly detection."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("ops-vision")
except PackageNotFoundError:
    __version__ = "1.0.0"

__author__: str = "Reflective Lantern"
__description__: str = (
    "SRE ML platform for real-time incident prediction, "
    "alert classification, and performance anomaly detection."
)

__all__ = ["__author__", "__description__", "__version__"]
