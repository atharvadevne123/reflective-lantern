"""Tests for fleet-mind app/geospatial_utils.py."""

from __future__ import annotations

import pytest

from app.geospatial_utils import (
    Coordinate,
    bearing_degrees,
    bounding_box,
    haversine_km,
    midpoint,
    route_distance_km,
)


class TestHaversineKm:
    def test_same_point_zero_distance(self) -> None:
        a = Coordinate(51.5, -0.1)
        assert haversine_km(a, a) == pytest.approx(0.0, abs=0.001)

    def test_london_to_paris(self) -> None:
        london = Coordinate(51.5074, -0.1278)
        paris = Coordinate(48.8566, 2.3522)
        dist = haversine_km(london, paris)
        assert 330 < dist < 360

    def test_symmetry(self) -> None:
        a = Coordinate(40.0, -74.0)
        b = Coordinate(51.5, -0.1)
        assert haversine_km(a, b) == pytest.approx(haversine_km(b, a), rel=1e-6)


class TestRouteDistanceKm:
    def test_single_point_zero(self) -> None:
        assert route_distance_km([Coordinate(0.0, 0.0)]) == pytest.approx(0.0)

    def test_empty_zero(self) -> None:
        assert route_distance_km([]) == pytest.approx(0.0)

    def test_two_points(self) -> None:
        a = Coordinate(51.5074, -0.1278)
        b = Coordinate(48.8566, 2.3522)
        assert route_distance_km([a, b]) > 0.0


class TestBearingDegrees:
    def test_north_bearing(self) -> None:
        a = Coordinate(0.0, 0.0)
        b = Coordinate(10.0, 0.0)
        bearing = bearing_degrees(a, b)
        assert bearing == pytest.approx(0.0, abs=1.0)

    def test_east_bearing(self) -> None:
        a = Coordinate(0.0, 0.0)
        b = Coordinate(0.0, 10.0)
        bearing = bearing_degrees(a, b)
        assert bearing == pytest.approx(90.0, abs=1.0)

    def test_result_between_zero_and_360(self) -> None:
        a = Coordinate(51.5, -0.1)
        b = Coordinate(48.8, 2.3)
        assert 0.0 <= bearing_degrees(a, b) < 360.0


class TestMidpoint:
    def test_midpoint_between_equator_points(self) -> None:
        a = Coordinate(0.0, -10.0)
        b = Coordinate(0.0, 10.0)
        mid = midpoint(a, b)
        assert mid.lat == pytest.approx(0.0, abs=0.01)
        assert mid.lon == pytest.approx(0.0, abs=0.01)


class TestBoundingBox:
    def test_empty_returns_zeros(self) -> None:
        bb = bounding_box([])
        assert bb["min_lat"] == 0.0

    def test_single_point_all_equal(self) -> None:
        bb = bounding_box([Coordinate(10.0, 20.0)])
        assert bb["min_lat"] == bb["max_lat"] == pytest.approx(10.0)

    def test_bounding_box_spans(self) -> None:
        coords = [Coordinate(0.0, 0.0), Coordinate(10.0, 20.0)]
        bb = bounding_box(coords)
        assert bb["max_lat"] == pytest.approx(10.0)
        assert bb["max_lon"] == pytest.approx(20.0)
