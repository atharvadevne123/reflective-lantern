"""Geospatial utility functions for fleet route analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass


EARTH_RADIUS_KM = 6371.0


@dataclass
class Coordinate:
    lat: float
    lon: float


def haversine_km(a: Coordinate, b: Coordinate) -> float:
    """Return great-circle distance in km between two coordinates."""
    lat1, lon1 = math.radians(a.lat), math.radians(a.lon)
    lat2, lon2 = math.radians(b.lat), math.radians(b.lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def route_distance_km(waypoints: list[Coordinate]) -> float:
    """Return total route distance for an ordered list of waypoints."""
    if len(waypoints) < 2:
        return 0.0
    return sum(haversine_km(waypoints[i], waypoints[i + 1]) for i in range(len(waypoints) - 1))


def bearing_degrees(a: Coordinate, b: Coordinate) -> float:
    """Return initial bearing in degrees (0-360) from a to b."""
    lat1 = math.radians(a.lat)
    lat2 = math.radians(b.lat)
    dlon = math.radians(b.lon - a.lon)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def midpoint(a: Coordinate, b: Coordinate) -> Coordinate:
    """Return geographic midpoint between two coordinates."""
    lat1, lon1 = math.radians(a.lat), math.radians(a.lon)
    lat2, lon2 = math.radians(b.lat), math.radians(b.lon)
    bx = math.cos(lat2) * math.cos(lon2 - lon1)
    by = math.cos(lat2) * math.sin(lon2 - lon1)
    mid_lat = math.atan2(
        math.sin(lat1) + math.sin(lat2),
        math.sqrt((math.cos(lat1) + bx) ** 2 + by**2),
    )
    mid_lon = lon1 + math.atan2(by, math.cos(lat1) + bx)
    return Coordinate(lat=math.degrees(mid_lat), lon=math.degrees(mid_lon))


def bounding_box(coords: list[Coordinate]) -> dict[str, float]:
    """Return min/max lat/lon bounding box for a list of coordinates."""
    if not coords:
        return {"min_lat": 0.0, "max_lat": 0.0, "min_lon": 0.0, "max_lon": 0.0}
    return {
        "min_lat": min(c.lat for c in coords),
        "max_lat": max(c.lat for c in coords),
        "min_lon": min(c.lon for c in coords),
        "max_lon": max(c.lon for c in coords),
    }
