"""Tests for app.geo_utils module."""

from __future__ import annotations

import math

import pytest

from app.geo_utils import (
    BoundingBox,
    Coordinate,
    bearing,
    bounding_box_of,
    haversine,
    midpoint,
    nearest_neighbor,
    within_radius,
)

LONDON = Coordinate(51.5074, -0.1278)
PARIS = Coordinate(48.8566, 2.3522)
NEW_YORK = Coordinate(40.7128, -74.0060)
SYDNEY = Coordinate(-33.8688, 151.2093)


class TestCoordinate:
    def test_valid_coordinate(self):
        c = Coordinate(lat=45.0, lon=90.0)
        assert c.lat == 45.0 and c.lon == 90.0

    def test_invalid_lat_raises(self):
        with pytest.raises(ValueError, match="Latitude"):
            Coordinate(lat=91, lon=0)

    def test_invalid_lon_raises(self):
        with pytest.raises(ValueError, match="Longitude"):
            Coordinate(lat=0, lon=181)

    @pytest.mark.parametrize("lat,lon", [(-90, -180), (90, 180), (0, 0), (45.5, -120.3)])
    def test_boundary_values_accepted(self, lat, lon):
        c = Coordinate(lat=lat, lon=lon)
        assert c.lat == lat and c.lon == lon

    def test_str_representation(self):
        c = Coordinate(lat=51.5074, lon=-0.1278)
        assert "51.507400" in str(c)


class TestHaversine:
    def test_same_point_is_zero(self):
        assert haversine(LONDON, LONDON) == pytest.approx(0.0, abs=1e-9)

    def test_london_to_paris_approx(self):
        dist = haversine(LONDON, PARIS)
        assert 340 < dist < 345

    def test_symmetry(self):
        assert haversine(LONDON, PARIS) == pytest.approx(haversine(PARIS, LONDON))

    def test_intercontinental_distance(self):
        dist = haversine(LONDON, NEW_YORK)
        assert 5500 < dist < 5600

    def test_antipodal_max_distance(self):
        north = Coordinate(0, 0)
        south = Coordinate(0, 180)
        dist = haversine(north, south)
        assert dist == pytest.approx(math.pi * 6371.0, rel=0.01)


class TestBoundingBox:
    def test_contains_interior_point(self):
        bbox = BoundingBox(min_lat=40, max_lat=50, min_lon=-5, max_lon=5)
        assert bbox.contains(Coordinate(45, 0))

    def test_rejects_exterior_point(self):
        bbox = BoundingBox(min_lat=40, max_lat=50, min_lon=-5, max_lon=5)
        assert not bbox.contains(Coordinate(60, 0))

    def test_center(self):
        bbox = BoundingBox(min_lat=40, max_lat=50, min_lon=-10, max_lon=10)
        center = bbox.center
        assert center.lat == pytest.approx(45.0)
        assert center.lon == pytest.approx(0.0)

    def test_bounding_box_of_list(self):
        coords = [LONDON, PARIS, NEW_YORK]
        bbox = bounding_box_of(coords)
        assert bbox.min_lat == min(c.lat for c in coords)
        assert bbox.max_lon == max(c.lon for c in coords)

    def test_bounding_box_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            bounding_box_of([])


class TestNearestNeighbor:
    def test_finds_closest(self):
        result = nearest_neighbor(LONDON, [PARIS, NEW_YORK, SYDNEY])
        assert result == PARIS

    def test_empty_candidates_raises(self):
        with pytest.raises(ValueError, match="empty"):
            nearest_neighbor(LONDON, [])

    def test_single_candidate(self):
        assert nearest_neighbor(LONDON, [PARIS]) == PARIS


class TestMidpoint:
    def test_midpoint_near_equator(self):
        a = Coordinate(0, -10)
        b = Coordinate(0, 10)
        mid = midpoint(a, b)
        assert mid.lat == pytest.approx(0.0, abs=0.01)
        assert mid.lon == pytest.approx(0.0, abs=0.01)

    def test_midpoint_london_paris_is_between(self):
        mid = midpoint(LONDON, PARIS)
        assert LONDON.lat > mid.lat > PARIS.lat


class TestBearingAndRadius:
    LONDON = Coordinate(51.5074, -0.1278)
    NORTH = Coordinate(52.5074, -0.1278)

    def test_bearing_due_north(self):
        b = bearing(self.LONDON, self.NORTH)
        assert b == pytest.approx(0.0, abs=0.5)

    def test_bearing_in_range(self):
        b = bearing(LONDON, PARIS)
        assert 0 <= b < 360

    def test_within_radius_includes_close_point(self):
        close = Coordinate(51.51, -0.13)
        result = within_radius(LONDON, 5.0, [close, PARIS])
        assert close in result
        assert PARIS not in result

    def test_within_radius_empty_candidates(self):
        assert within_radius(LONDON, 100.0, []) == []

    def test_within_radius_all_included(self):
        nearby = [Coordinate(51.5, -0.12), Coordinate(51.52, -0.13)]
        result = within_radius(LONDON, 10.0, nearby)
        assert len(result) == 2
