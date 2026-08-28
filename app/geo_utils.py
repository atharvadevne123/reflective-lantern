"""Geographic utilities: distance, bounding box, and coordinate helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

_EARTH_RADIUS_KM = 6_371.0


@dataclass(frozen=True)
class Coordinate:
    """An immutable WGS-84 geographic coordinate.

    Attributes:
        lat: Latitude in decimal degrees (-90 to 90).
        lon: Longitude in decimal degrees (-180 to 180).
    """

    lat: float
    lon: float

    def __post_init__(self) -> None:
        if not -90 <= self.lat <= 90:
            raise ValueError(f"Latitude must be in [-90, 90], got {self.lat}")
        if not -180 <= self.lon <= 180:
            raise ValueError(f"Longitude must be in [-180, 180], got {self.lon}")

    def __str__(self) -> str:
        return f"({self.lat:.6f}, {self.lon:.6f})"


def haversine(a: Coordinate, b: Coordinate) -> float:
    """Compute the haversine great-circle distance between two coordinates.

    Args:
        a: First coordinate.
        b: Second coordinate.

    Returns:
        Distance in kilometres.
    """
    lat1, lon1 = math.radians(a.lat), math.radians(a.lon)
    lat2, lon2 = math.radians(b.lat), math.radians(b.lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(h))


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned bounding box in geographic coordinates.

    Attributes:
        min_lat: Southern latitude bound.
        max_lat: Northern latitude bound.
        min_lon: Western longitude bound.
        max_lon: Eastern longitude bound.
    """

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def contains(self, coord: Coordinate) -> bool:
        """Return True if coord falls within this bounding box."""
        return (
            self.min_lat <= coord.lat <= self.max_lat
            and self.min_lon <= coord.lon <= self.max_lon
        )

    @property
    def center(self) -> Coordinate:
        """Return the geographic center of the bounding box."""
        return Coordinate(
            lat=(self.min_lat + self.max_lat) / 2,
            lon=(self.min_lon + self.max_lon) / 2,
        )


def bounding_box_of(coords: List[Coordinate]) -> BoundingBox:
    """Compute the smallest bounding box that contains all coordinates.

    Args:
        coords: Non-empty list of coordinates.

    Returns:
        :class:`BoundingBox` enclosing all points.

    Raises:
        ValueError: If coords is empty.
    """
    if not coords:
        raise ValueError("coords must not be empty")
    return BoundingBox(
        min_lat=min(c.lat for c in coords),
        max_lat=max(c.lat for c in coords),
        min_lon=min(c.lon for c in coords),
        max_lon=max(c.lon for c in coords),
    )


def nearest_neighbor(query: Coordinate, candidates: List[Coordinate]) -> Coordinate:
    """Return the candidate closest to query using haversine distance.

    Args:
        query: The reference coordinate.
        candidates: Non-empty list of candidates.

    Returns:
        Closest :class:`Coordinate`.

    Raises:
        ValueError: If candidates is empty.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")
    return min(candidates, key=lambda c: haversine(query, c))


def midpoint(a: Coordinate, b: Coordinate) -> Coordinate:
    """Compute the geographic midpoint between two coordinates.

    Uses the spherical midpoint formula for accuracy over large distances.

    Args:
        a: First coordinate.
        b: Second coordinate.

    Returns:
        Midpoint :class:`Coordinate`.
    """
    lat1 = math.radians(a.lat)
    lat2 = math.radians(b.lat)
    dlon = math.radians(b.lon - a.lon)
    bx = math.cos(lat2) * math.cos(dlon)
    by = math.cos(lat2) * math.sin(dlon)
    mid_lat = math.degrees(math.atan2(
        math.sin(lat1) + math.sin(lat2),
        math.sqrt((math.cos(lat1) + bx) ** 2 + by ** 2),
    ))
    mid_lon = a.lon + math.degrees(math.atan2(by, math.cos(lat1) + bx))
    return Coordinate(lat=mid_lat, lon=mid_lon)


__all__ = [
    "BoundingBox",
    "Coordinate",
    "bounding_box_of",
    "haversine",
    "midpoint",
    "nearest_neighbor",
]
