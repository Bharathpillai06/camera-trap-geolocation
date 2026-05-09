"""Unit tests for the geometry module.

Run with::

    pip install pytest
    pytest tests/
"""

import math

import pytest

from camera_trap_geolocation.geometry import (
    estimate_distance_and_angle,
    focal_length_px,
    parse_camera_bearing,
    project_gps,
)



def test_focal_length_60deg_hfov_1920():
    """At 60 deg HFOV, focal length should be (W/2) / tan(30 deg) = W/2 * sqrt(3)."""
    fx = focal_length_px(1920, 60.0)
    expected = 960.0 / math.tan(math.radians(30.0))
    assert fx == pytest.approx(expected, rel=1e-9)


def test_focal_length_90deg_hfov():
    """At 90 deg HFOV, fx = W/2 (since tan(45) = 1)."""
    fx = focal_length_px(2000, 90.0)
    assert fx == pytest.approx(1000.0, rel=1e-9)




def test_distance_centered_bbox():
    """A bbox centered horizontally has angle ~0; distance follows from bbox height."""
    img_w, img_h = 1920, 1080
    x_min, y_min, x_max, y_max = 810, 400, 1110, 700
    d_m, a_deg = estimate_distance_and_angle(
        img_w, img_h, x_min, y_min, x_max, y_max,
        hfov_deg=60.0, real_height_m=0.9,
    )
    fx = focal_length_px(img_w, 60.0)
    expected_d = (0.9 * fx) / 300.0
    assert d_m == pytest.approx(expected_d, rel=1e-9)
    assert a_deg == pytest.approx(0.0, abs=1e-9)


def test_angle_off_center_positive():
    """A bbox whose centroid is to the right of center yields a positive angle."""
    _, a_deg = estimate_distance_and_angle(
        1920, 1080, 1500, 400, 1700, 700,
        hfov_deg=60.0, real_height_m=0.9,
    )
    assert a_deg > 0


def test_angle_off_center_negative():
    """A bbox whose centroid is to the left of center yields a negative angle."""
    _, a_deg = estimate_distance_and_angle(
        1920, 1080, 200, 400, 400, 700,
        hfov_deg=60.0, real_height_m=0.9,
    )
    assert a_deg < 0


def test_invalid_bbox_returns_inf():
    """A bbox with non-positive height should return inf distance, 0 angle."""
    d, a = estimate_distance_and_angle(
        1920, 1080, 100, 500, 200, 500,
        hfov_deg=60.0, real_height_m=0.9,
    )
    assert d == float("inf")
    assert a == 0.0



def test_project_zero_distance_returns_input():
    lat, lon = project_gps(40.0, -83.0, 0.0, 90.0)
    assert lat == pytest.approx(40.0, abs=1e-9)
    assert lon == pytest.approx(-83.0, abs=1e-9)


def test_project_north_increases_latitude():
    """Projecting 1 km due north should increase latitude by ~0.009 deg."""
    lat, lon = project_gps(40.0, -83.0, 1000.0, 0.0)
    assert lat > 40.0
    assert lat == pytest.approx(40.0 + 1000.0 / 111_320.0, rel=1e-3)
    assert lon == pytest.approx(-83.0, abs=1e-6)


def test_project_east_increases_longitude():
    """Projecting east should increase longitude (smaller change at higher latitude)."""
    lat, lon = project_gps(40.0, -83.0, 1000.0, 90.0)
    assert lon > -83.0
    # Approx 1000 m east at lat=40 is ~1000 / (111320 * cos(40 deg)) deg
    expected_dlon = 1000.0 / (111_320.0 * math.cos(math.radians(40.0)))
    assert (lon + 83.0) == pytest.approx(expected_dlon, rel=1e-2)


def test_project_round_trip_180():
    """Going 100 m north then 100 m south should return roughly to start."""
    lat1, lon1 = project_gps(40.0, -83.0, 100.0, 0.0)
    lat2, lon2 = project_gps(lat1, lon1, 100.0, 180.0)
    assert lat2 == pytest.approx(40.0, abs=1e-6)
    assert lon2 == pytest.approx(-83.0, abs=1e-6)



@pytest.mark.parametrize("val", [None, "", "NA", "na", "  NA  "])
def test_parse_bearing_na_or_empty_returns_none(val):
    assert parse_camera_bearing(val) is None


def test_parse_bearing_numeric_string():
    assert parse_camera_bearing("90") == 90.0


def test_parse_bearing_number():
    assert parse_camera_bearing(180.5) == 180.5


def test_parse_bearing_wraps_to_0_360():
    assert parse_camera_bearing(370.5) == pytest.approx(10.5)
    assert parse_camera_bearing(-10) == pytest.approx(350.0)


def test_parse_bearing_garbage_returns_none():
    assert parse_camera_bearing("not a number") is None
